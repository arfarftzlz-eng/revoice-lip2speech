"""
USR 2.0 — Single-video inference demo.

Usage
-----
Audio-visual (default):
    python demo.py --video path/to/video.mp4 --checkpoint path/to/model.pth

Video-only:
    python demo.py --video path/to/video.mp4 --checkpoint path/to/model.pth --modality video

Audio-only:
    python demo.py --video path/to/video.mp4 --checkpoint path/to/model.pth --modality audio

Override backbone:
    python demo.py --video path/to/video.mp4 --checkpoint path/to/model.pth \
        model/backbone=resnet_transformer_large

Any additional Hydra overrides can be appended after the script arguments.
"""

import logging
import sys

import cv2
import hydra
import numpy as np
import torch
import torchaudio
import torchvision
from omegaconf import DictConfig, OmegaConf
from torchvision.transforms import CenterCrop, Compose, Grayscale, Lambda

from data.transforms import NormalizeVideo
from espnet.asr.asr_utils import parse_hypothesis
from espnet.nets.batch_beam_search import BatchBeamSearch
from espnet.nets.pytorch_backend.e2e_asr_transformer import E2E
from espnet.nets.scorers.length_bonus import LengthBonus
from language_reranker import RecognitionCandidate
from preprocessing.landmarks_detector import LandmarksDetector
from preprocessing.video_preprocess import VideoProcess
from prosody import RecognitionResult, build_recognition_result
from utils.utils import UNIGRAM1000_LIST

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Video / audio loading
# ---------------------------------------------------------------------------

def load_video_audio(path: str, target_fps: int = 25, target_sr: int = 16000):
    """Load a video file and return (video_frames, audio_waveform).

    Returns
    -------
    video : np.ndarray, uint8, (T, H, W, C) RGB
    audio : torch.Tensor, float32, (1, S)  mono waveform at ``target_sr``
    """
    import os
    if not os.path.exists(path):
        raise FileNotFoundError(f"Video file not found: {path}")
    video, audio, info = torchvision.io.read_video(path, pts_unit="sec")
    # video: (T, H, W, C) uint8   audio: (C, S) float

    # --- audio ---------------------------------------------------------------
    if audio.numel() == 0:
        # Synthesise silence matching video duration
        n_samples = int(video.shape[0] / target_fps * target_sr)
        audio = torch.zeros(1, n_samples)
    else:
        audio = audio.mean(dim=0, keepdim=True)  # stereo -> mono
        if int(info["audio_fps"]) != target_sr:
            audio = torchaudio.transforms.Resample(
                int(info["audio_fps"]), target_sr
            )(audio)

    # --- video ---------------------------------------------------------------
    # Handle different torchvision versions and video formats
    vfps = info.get("video_fps") or info.get("fps")
    if not vfps:
        log.warning(
            "Could not determine video FPS from metadata. Assuming %d FPS. "
            "If your video has a different frame rate, results may be degraded. "
            "Consider re-encoding: ffmpeg -i input.mp4 -r 25 -ar 16000 output.mp4",
            target_fps
        )
        vfps = target_fps
    if abs(vfps - target_fps) > 1e-3:
        n_frames = video.shape[0]
        new_n = int(n_frames / vfps * target_fps)
        indices = torch.linspace(0, n_frames - 1, new_n).long()
        video = video[indices]

    # align audio length to video
    expected_samples = video.shape[0] * (target_sr // target_fps)
    if audio.shape[1] < expected_samples:
        audio = torch.nn.functional.pad(audio, (0, expected_samples - audio.shape[1]))
    else:
        audio = audio[:, :expected_samples]

    video = video.numpy()  # (T, H, W, C) uint8
    return video, audio


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def build_video_transform():
    """Mouth-crop -> normalised grayscale tensor (T, H, W)."""
    return Compose([
        Lambda(lambda x: x / 255.0),
        CenterCrop(88),
        Lambda(lambda x: x.transpose(0, 1)),  # (C,T,H,W) -> (T,C,H,W) for Grayscale
        Grayscale(),
        Lambda(lambda x: x.transpose(0, 1)),  # (T,1,H,W) -> (1,T,H,W) for NormalizeVideo
        NormalizeVideo(mean=(0.421,), std=(0.165,)),
        Lambda(lambda x: x.squeeze(0)),       # (1,T,H,W) -> (T,H,W)
    ])


def save_mouth_crop(mouth_video: np.ndarray, output_path: str = "mouth_crop.mp4", fps: int = 25):
    """Save the raw mouth crop as a video file for inspection."""
    T, H, W, C = mouth_video.shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (W, H))
    for t in range(T):
        frame = cv2.cvtColor(mouth_video[t], cv2.COLOR_RGB2BGR)
        writer.write(frame)
    writer.release()
    log.info("Saved mouth crop to: %s (%d frames, %dx%d)", output_path, T, W, H)


def preprocess_video(
    video_frames: np.ndarray,
    landmarks_detector,
    video_processor,
    mouth_crop_path: str = "mouth_crop.mp4",
    return_mouth_video: bool = False,
):
    """Detect landmarks and return the normalized mouth tensor.

    ``return_mouth_video`` preserves the stabilized RGB mouth crop for visual
    pause analysis without changing the original inference API.
    """
    landmarks = landmarks_detector(video_frames)
    mouth_video = video_processor(video_frames, landmarks)
    if mouth_video is None:
        raise RuntimeError(
            "Could not detect a face in enough frames. "
            "Make sure the video contains a clearly visible face."
        )
    save_mouth_crop(mouth_video, mouth_crop_path)
    # (T, H, W, C) -> (C, T, H, W) float tensor
    video_tensor = torch.from_numpy(mouth_video).permute(3, 0, 1, 2).float()
    video_tensor = build_video_transform()(video_tensor)
    if return_mouth_video:
        return video_tensor, mouth_video
    return video_tensor  # (T, H, W)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(cfg: DictConfig, checkpoint_path: str, device: torch.device):
    """Instantiate E2E model, load weights, set to eval."""
    model = E2E(len(UNIGRAM1000_LIST), cfg.model.backbone)

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    # Strip torch.compile prefix if present
    if any(k.startswith("_orig_mod.") for k in ckpt):
        ckpt = {k.replace("_orig_mod.", "", 1): v for k, v in ckpt.items()}
    # Strip wrapper module prefix if present
    if any(k.startswith("model.backbone.") for k in ckpt):
        ckpt = {
            k.replace("model.backbone.", "", 1): v
            for k, v in ckpt.items()
            if k.startswith("model.backbone.")
        }
    model.load_state_dict(ckpt)
    model.eval().to(device)
    return model


# ---------------------------------------------------------------------------
# Beam search
# ---------------------------------------------------------------------------

def build_beam_search(cfg: DictConfig, model: E2E):
    """Build a BatchBeamSearch scorer."""
    token_list = UNIGRAM1000_LIST
    scorers = model.scorers()
    scorers["length_bonus"] = LengthBonus(len(token_list))
    weights = dict(
        decoder=1.0 - cfg.decode.ctc_weight,
        ctc=cfg.decode.ctc_weight,
        length_bonus=cfg.decode.penalty,
    )
    return BatchBeamSearch(
        beam_size=cfg.decode.beam_size,
        vocab_size=len(token_list),
        weights=weights,
        scorers=scorers,
        sos=len(token_list) - 1,
        eos=len(token_list) - 1,
        token_list=token_list,
        pre_beam_score_key=None if cfg.decode.ctc_weight == 1.0 else "decoder",
    )


# ---------------------------------------------------------------------------
# Decoding
# ---------------------------------------------------------------------------

def _candidate_from_hypothesis(hypothesis, rank: int) -> RecognitionCandidate:
    """Keep text, tokens and component scores for safe N-best reranking."""
    data = hypothesis.asdict()
    text, _, token_id_text, _ = parse_hypothesis(data, UNIGRAM1000_LIST)
    text = text.replace("<eos>", "").replace("\u2581", " ").strip()
    token_ids = [int(value) for value in token_id_text.split()]
    eos_id = len(UNIGRAM1000_LIST) - 1
    token_ids = [token_id for token_id in token_ids if token_id not in (0, eos_id)]
    return RecognitionCandidate(
        rank=rank,
        text=text,
        token_ids=tuple(token_ids),
        score=float(data.get("score", 0.0)),
        component_scores={
            str(name): float(score)
            for name, score in (data.get("scores") or {}).items()
        },
    )


def decode_nbest(
    features: torch.Tensor,
    beam_search: BatchBeamSearch,
    modality: str,
    cfg: DictConfig,
    max_candidates: int = 5,
) -> list[RecognitionCandidate]:
    """Run beam search once and retain unique, unchanged N-best candidates."""
    if max_candidates < 1:
        raise ValueError("max_candidates must be at least 1")
    hyps = beam_search(
        x=features.squeeze(0),
        modality=modality,
        maxlenratio=cfg.decode.maxlenratio,
        minlenratio=cfg.decode.minlenratio,
    )
    candidates = []
    seen = set()
    for hypothesis in hyps:
        candidate = _candidate_from_hypothesis(hypothesis, len(candidates) + 1)
        normalized_text = " ".join(candidate.text.upper().split())
        if not normalized_text or normalized_text in seen:
            continue
        seen.add(normalized_text)
        candidates.append(candidate)
        if len(candidates) >= max_candidates:
            break
    if not candidates:
        raise RuntimeError("Beam search returned no non-empty hypotheses")
    return candidates


def _decode_hypothesis(features: torch.Tensor, beam_search: BatchBeamSearch,
                       modality: str, cfg: DictConfig):
    """Run beam search once and keep both display text and token IDs."""
    best = decode_nbest(features, beam_search, modality, cfg, max_candidates=1)[0]
    return best.text, list(best.token_ids)


def decode(features: torch.Tensor, beam_search: BatchBeamSearch,
           modality: str, cfg: DictConfig) -> str:
    """Run beam search and return the 1-best transcription string."""
    text, _ = _decode_hypothesis(features, beam_search, modality, cfg)
    return text


def decode_with_timing(
    features: torch.Tensor,
    beam_search: BatchBeamSearch,
    modality: str,
    cfg: DictConfig,
    *,
    ctc_module,
    mouth_video: np.ndarray,
    video_duration_s: float,
) -> RecognitionResult:
    """Decode text and conservatively recover CTC/visual pause timing."""
    text, token_ids = _decode_hypothesis(features, beam_search, modality, cfg)
    if not text or not token_ids or ctc_module is None:
        return RecognitionResult(
            text=text,
            token_ids=token_ids,
            tokens=[UNIGRAM1000_LIST[token_id] for token_id in token_ids],
            video_duration_s=video_duration_s,
            alignment_status="unavailable: no CTC-aligned text",
        )
    log_probs = ctc_module.log_softmax(features).squeeze(0)
    return build_recognition_result(
        text=text,
        token_ids=token_ids,
        token_list=UNIGRAM1000_LIST,
        log_probs=log_probs,
        mouth_video=mouth_video,
        video_duration_s=video_duration_s,
    )


@torch.no_grad()
def transcribe(video_path: str, cfg: DictConfig, modality: str = "av",
               device: torch.device = torch.device("cuda"),
               detector: str = "mediapipe"):
    """End-to-end: video path in, transcription string out."""
    log.info("Loading video: %s", video_path)
    video_frames, audio = load_video_audio(video_path)

    # --- face / mouth preprocessing -----------------------------------------
    log.info("Detecting landmarks and cropping mouth region ...")
    ld = LandmarksDetector(detector=detector)
    vp = VideoProcess(convert_gray=False)
    video_tensor = preprocess_video(video_frames, ld, vp)
    ld.close()  # Explicitly close to avoid Python 3.13+ shutdown errors

    # --- model ---------------------------------------------------------------
    log.info("Loading model from: %s", cfg.model.pretrained_model_path)
    model = load_model(cfg, cfg.model.pretrained_model_path, device)
    beam_search = build_beam_search(cfg, model)
    beam_search.to(device)

    # --- encode --------------------------------------------------------------
    video_input = video_tensor.unsqueeze(0).to(device)       # (1, T, H, W)
    audio_input = audio.unsqueeze(0).to(device).transpose(1, 2)  # (1, S, 1) -> (1, 1, S)

    modality_key = modality[0].lower() if modality not in ("a", "v", "av") else modality
    if modality_key == "av":
        feat = model.encoder(xs_v=video_input, xs_a=audio_input)
    elif modality_key == "v":
        feat = model.encoder(xs_v=video_input)
    elif modality_key == "a":
        feat = model.encoder(xs_a=audio_input)
    else:
        raise ValueError(f"Unknown modality '{modality}'. Choose from: av, video (v), audio (a).")

    # --- decode --------------------------------------------------------------
    log.info("Decoding with modality=%s, beam_size=%d ...", modality_key, cfg.decode.beam_size)
    text = decode(feat, beam_search, modality_key, cfg)
    return text


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

# Register custom resolvers so Hydra config doesn't fail on missing model keys
OmegaConf.register_new_resolver("len", len, replace=True)


@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    # Pull --video and --modality from Hydra overrides (or config)
    video_path = cfg.get("video")
    checkpoint = cfg.get("model", {}).get("pretrained_model_path")
    modality = cfg.get("modality", "av")

    if not video_path:
        print("Error: --video is required. Usage:")
        print("  python demo.py video=path/to/video.mp4 model.pretrained_model_path=path/to/model.pth")
        sys.exit(1)
    if not checkpoint:
        print("Error: model.pretrained_model_path is required.")
        sys.exit(1)

    detector = cfg.get("detector", "mediapipe")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    text = transcribe(video_path, cfg, modality=modality, device=device,
                      detector=detector)
    print(f"\n{'='*60}")
    print(f" Modality : {modality}")
    print(f" Video    : {video_path}")
    print(f" Result   : {text}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
