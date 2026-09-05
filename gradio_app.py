"""Gradio frontend for USR 2.0 visual speech recognition.

The camera tab implements near-real-time *clip* recognition: the browser
records a short video, then the complete clip is passed to the same offline
sequence recognizer used for uploaded files. It is intentionally not described
as streaming VSR because USR 2.0 decodes complete sequences.
"""

# Environment variables intentionally must be set before third-party imports.
# ruff: noqa: E402

import atexit
import logging
import os
import shutil
import tempfile
import threading
import time
from collections import OrderedDict, deque
from pathlib import Path
from typing import Optional, Tuple

# Must be configured before importing torch. An explicit environment value
# from the launch command still takes precedence.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")


def _find_bundled_ffmpeg() -> Optional[Path]:
    if shutil.which("ffmpeg"):
        return None
    try:
        import imageio_ffmpeg
    except ImportError:
        return None
    ffmpeg_path = Path(imageio_ffmpeg.get_ffmpeg_exe())
    os.environ["PATH"] = str(ffmpeg_path.parent) + os.pathsep + os.environ.get("PATH", "")
    return ffmpeg_path


def _exclude_localhost_from_proxy() -> None:
    """Keep Gradio's local startup health check away from global proxies."""
    required = {"127.0.0.1", "localhost", "::1"}
    for key in ("NO_PROXY", "no_proxy"):
        current = {item.strip() for item in os.environ.get(key, "").split(",") if item.strip()}
        os.environ[key] = ",".join(sorted(current | required))


_BUNDLED_FFMPEG = _find_bundled_ffmpeg()
_exclude_localhost_from_proxy()

import gradio as gr
import torch
from gradio._vendor.ffmpy import FFmpeg as GradioFFmpeg
from gradio.components import video as gradio_video
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from demo import (
    build_beam_search,
    decode_nbest,
    decode_with_timing,
    load_model,
    load_video_audio,
    preprocess_video,
)
from language_reranker import (
    DEFAULT_RERANK_API_BASE,
    DEFAULT_RERANK_MODEL,
    RerankConfig,
    RerankDecision,
    format_rerank_markdown,
    rerank_candidates,
    test_language_connection,
)
from preprocessing.landmarks_detector import LandmarksDetector
from preprocessing.video_preprocess import VideoProcess
from prosody import RecognitionResult, build_recognition_result, format_timeline
from tts import (
    MINIMAX_REGION_AUTO,
    MINIMAX_REGION_CHOICES,
    MODE_AUTO,
    STYLE_LEVEL_CHOICES,
    STYLE_NATURAL,
    TTS_MODE_CHOICES,
    synthesize_speech,
    validate_minimax_key,
)
from utils.utils import UNIGRAM1000_LIST


if _BUNDLED_FFMPEG is not None:
    class BundledFFmpeg(GradioFFmpeg):
        def __init__(self, *args, **kwargs):
            kwargs["executable"] = str(_BUNDLED_FFMPEG)
            super().__init__(*args, **kwargs)

    gradio_video.FFmpeg = BundledFFmpeg


LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = Path(
    PROJECT_ROOT / "checkpoint" / "huge_high_resource_lrs2lrs3vox2avsp.pth"
)
BACKBONE_OVERRIDE = "model/backbone=resnet_transformer_huge"
MIN_VIDEO_FRAMES = 12
MAX_SAVED_RESULTS = 20
MAX_BEAM_CACHE_SIZE = 2
GPU_CONCURRENCY_ID = "usr2_gpu_inference"
TTS_CONCURRENCY_ID = "usr2_speech_synthesis"


def _safe_beam_limit_for_vram(total_vram_gib: float) -> int:
    """Choose a conservative decoder limit for the installed GPU."""
    if total_vram_gib < 12:
        return 5
    if total_vram_gib < 18:
        return 10
    if total_vram_gib < 28:
        return 20
    return 40


def _is_cuda_oom(error: BaseException) -> bool:
    """Recognize both PyTorch OOM classes and ESPnet's RuntimeError form."""
    if isinstance(error, torch.cuda.OutOfMemoryError):
        return True
    message = str(error).lower()
    return "out of memory" in message and ("cuda" in message or "cublas" in message)


def _safe_cuda_cleanup() -> bool:
    """Release cached blocks without allowing a poisoned CUDA stream to escape."""
    if not torch.cuda.is_available():
        return True
    cleanup_ok = True
    try:
        torch.cuda.synchronize()
    except Exception:
        cleanup_ok = False
        LOGGER.debug("CUDA synchronization failed during cleanup", exc_info=True)
    try:
        torch.cuda.empty_cache()
    except Exception:
        cleanup_ok = False
        LOGGER.debug("CUDA cache cleanup failed", exc_info=True)
    return cleanup_ok


def build_config(checkpoint_path: Path = DEFAULT_CHECKPOINT) -> DictConfig:
    """Compose the same Hydra configuration used by ``demo.py``."""
    with initialize_config_dir(
        config_dir=str(PROJECT_ROOT / "conf"), version_base="1.3"
    ):
        cfg = compose(config_name="config", overrides=[BACKBONE_OVERRIDE])
    cfg.model.pretrained_model_path = str(checkpoint_path)
    cfg.modality = "v"
    cfg.detector = "mediapipe"
    return cfg


class ResultDirectoryManager:
    """Keep recent ROI videos available to Gradio without leaking temp files."""

    def __init__(self, max_results: int = MAX_SAVED_RESULTS):
        self.root = Path(tempfile.mkdtemp(prefix="usr2_gradio_results_"))
        self.max_results = max_results
        self._results = deque()
        self._lock = threading.Lock()
        atexit.register(self.close)

    def create(self) -> Path:
        with self._lock:
            while len(self._results) >= self.max_results:
                shutil.rmtree(self._results.popleft(), ignore_errors=True)
            result_dir = Path(tempfile.mkdtemp(prefix="request_", dir=self.root))
            self._results.append(result_dir)
            return result_dir

    def discard(self, result_dir: Path) -> None:
        with self._lock:
            try:
                self._results.remove(result_dir)
            except ValueError:
                pass
        shutil.rmtree(result_dir, ignore_errors=True)

    def close(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)


class VSRService:
    """Own one Huge model and serialize all GPU inference requests."""

    def __init__(self, checkpoint_path: Path = DEFAULT_CHECKPOINT):
        self.checkpoint_path = Path(checkpoint_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.total_vram_gib = 0.0
        self.safe_beam_limit = 40
        if self.device.type == "cuda":
            properties = torch.cuda.get_device_properties(self.device)
            self.total_vram_gib = properties.total_memory / (1024 ** 3)
            self.safe_beam_limit = _safe_beam_limit_for_vram(self.total_vram_gib)
        self.cfg: Optional[DictConfig] = None
        self.model = None
        self.startup_error = ""
        self.results = ResultDirectoryManager()
        self._inference_lock = threading.Lock()
        self._detectors = {}
        self._beam_searches = OrderedDict()
        self._video_processor = VideoProcess(convert_gray=False)
        self._initialize_model()
        atexit.register(self.close)

    def _initialize_model(self) -> None:
        try:
            if not self.checkpoint_path.is_file():
                raise FileNotFoundError(
                    f"Checkpoint not found: {self.checkpoint_path}"
                )
            self.cfg = build_config(self.checkpoint_path)
            LOGGER.info(
                "Loading USR 2.0 Huge model once on %s from %s",
                self.device,
                self.checkpoint_path,
            )
            with torch.inference_mode():
                self.model = load_model(self.cfg, str(self.checkpoint_path), self.device)
            # Warm the GPU-safe decoder rather than the old unconditional Beam 40.
            default_key = (self.safe_beam_limit, 0.1)
            self._beam_searches[default_key] = self._make_beam_search(*default_key)
            LOGGER.info("USR 2.0 model is ready")
        except Exception as error:  # Keep the UI alive with an actionable status.
            self.startup_error = f"Model initialization failed: {error}"
            self.model = None
            LOGGER.exception(self.startup_error)
            _safe_cuda_cleanup()

    @property
    def status_message(self) -> str:
        if self.startup_error:
            return f"❌ **Model unavailable:** {self.startup_error}"
        if self.device.type == "cuda":
            gpu = torch.cuda.get_device_name(self.device)
            return (
                f"✅ **Model ready** · USR 2.0 Huge · `{gpu}` · "
                f"{self.total_vram_gib:.1f} GB VRAM · safe Beam ≤ {self.safe_beam_limit}"
            )
        return "✅ **Model ready** · USR 2.0 Huge · `CPU`"

    def _request_config(self, beam_size: int, ctc_weight: float) -> DictConfig:
        cfg = OmegaConf.merge(
            self.cfg,
            {"decode": {"beam_size": beam_size, "ctc_weight": ctc_weight}},
        )
        return cfg

    def _make_beam_search(self, beam_size: int, ctc_weight: float):
        cfg = self._request_config(beam_size, ctc_weight)
        beam_search = build_beam_search(cfg, self.model)
        beam_search.to(self.device)
        return beam_search

    def _get_beam_search(self, beam_size: int, ctc_weight: float):
        key = (beam_size, round(ctc_weight, 4))
        if key in self._beam_searches:
            self._beam_searches.move_to_end(key)
            return self._beam_searches[key]
        self._beam_searches[key] = self._make_beam_search(*key)
        while len(self._beam_searches) > MAX_BEAM_CACHE_SIZE:
            self._beam_searches.popitem(last=False)
        return self._beam_searches[key]

    def _get_detector(self, detector_name: str) -> LandmarksDetector:
        if detector_name not in {"mediapipe", "retinaface"}:
            raise ValueError(f"Unsupported detector: {detector_name}")
        if detector_name not in self._detectors:
            self._detectors[detector_name] = LandmarksDetector(
                detector=detector_name, device=str(self.device)
            )
        return self._detectors[detector_name]

    @torch.inference_mode()
    def _recognize_locked(
        self,
        video_path: str,
        detector_name: str,
        beam_size: int,
        ctc_weight: float,
        mouth_crop_path: Path,
        rerank_config: Optional[RerankConfig] = None,
    ) -> tuple[RecognitionResult, Optional[RerankDecision]]:
        video_frames, _ = load_video_audio(video_path)
        frame_count = int(video_frames.shape[0])
        if frame_count == 0:
            raise ValueError("The video contains no readable frames.")
        if frame_count < MIN_VIDEO_FRAMES:
            raise ValueError(
                f"The clip is too short ({frame_count} frames after conversion to "
                f"25 FPS). Record at least 0.5 seconds; 2–5 seconds is recommended."
            )

        detector = self._get_detector(detector_name)
        video_tensor, mouth_video = preprocess_video(
            video_frames,
            detector,
            self._video_processor,
            mouth_crop_path=str(mouth_crop_path),
            return_mouth_video=True,
        )
        request_cfg = self._request_config(beam_size, ctc_weight)
        beam_search = self._get_beam_search(beam_size, ctc_weight)
        video_input = video_tensor.unsqueeze(0).to(self.device)
        features = self.model.encoder(xs_v=video_input)
        rerank_decision = None
        if rerank_config is not None and rerank_config.enabled and beam_size > 1:
            candidates = decode_nbest(
                features,
                beam_search,
                "v",
                request_cfg,
                max_candidates=min(5, beam_size),
            )
            rerank_decision = rerank_candidates(candidates, rerank_config)
            selected = rerank_decision.selected
            log_probs = self.model.ctc_v.log_softmax(features).squeeze(0)
            result = build_recognition_result(
                text=selected.text,
                token_ids=list(selected.token_ids),
                token_list=UNIGRAM1000_LIST,
                log_probs=log_probs,
                mouth_video=mouth_video,
                video_duration_s=frame_count / 25.0,
            )
        else:
            result = decode_with_timing(
                features,
                beam_search,
                "v",
                request_cfg,
                ctc_module=self.model.ctc_v,
                mouth_video=mouth_video,
                video_duration_s=frame_count / 25.0,
            )
        return result, rerank_decision

    def recognize(
        self,
        video_path: Optional[str],
        detector_name: str,
        beam_size: float,
        ctc_weight: float,
        auto_read_aloud: bool,
        tts_mode: str,
        expression_level: str,
        preserve_timing: bool,
        minimax_region: str = MINIMAX_REGION_AUTO,
        minimax_api_key: str = "",
        minimax_api_base: str = "",
        rerank_enabled: bool = False,
        rerank_api_base: str = DEFAULT_RERANK_API_BASE,
        rerank_model: str = DEFAULT_RERANK_MODEL,
        rerank_api_key: str = "",
    ) -> tuple:
        """Run VSR once and optionally synthesize its complete result."""
        started = time.perf_counter()
        result_dir = None
        if not video_path:
            return (
                None,
                "",
                None,
                format_timeline(None),
                None,
                "0.00 s",
                "Please upload or record a video first.",
                "",
                "ℹ️ **Language reranking:** No video was submitted.",
            )
        if self.model is None:
            return (
                None,
                "",
                None,
                format_timeline(None),
                None,
                "0.00 s",
                self.startup_error,
                "",
                "ℹ️ **Language reranking:** Recognition model unavailable.",
            )

        try:
            requested_beam_size = int(beam_size)
            ctc_weight = float(ctc_weight)
            if not 1 <= requested_beam_size <= 40:
                raise ValueError("Beam size must be between 1 and 40.")
            if not 0.0 <= ctc_weight <= 1.0:
                raise ValueError("CTC weight must be between 0 and 1.")
            rerank_config = RerankConfig(
                enabled=bool(rerank_enabled),
                api_key=rerank_api_key,
                api_base=rerank_api_base or DEFAULT_RERANK_API_BASE,
                model=rerank_model or DEFAULT_RERANK_MODEL,
            )
            beam_size = min(requested_beam_size, self.safe_beam_limit)
            recognition_guidance = ""
            if beam_size != requested_beam_size:
                recognition_guidance = (
                    f"GPU memory guard: Beam {requested_beam_size} was reduced to "
                    f"Beam {beam_size} for this {self.total_vram_gib:.1f} GB GPU."
                )
            input_path = Path(video_path)
            if not input_path.is_file():
                raise FileNotFoundError(f"Input video not found: {input_path}")

            result_dir = self.results.create()
            mouth_crop_path = result_dir / "mouth_crop.mp4"
            # The Gradio queue already limits concurrency, while this lock also
            # protects direct API calls and accidental queue bypasses.
            with self._inference_lock:
                try:
                    result, rerank_decision = self._recognize_locked(
                        str(input_path),
                        detector_name,
                        beam_size,
                        ctc_weight,
                        mouth_crop_path,
                        rerank_config,
                    )
                except Exception as error:
                    if not _is_cuda_oom(error) or beam_size == 1:
                        raise
                    LOGGER.warning(
                        "CUDA OOM at Beam %d; retrying once at Beam 1", beam_size
                    )
                    self._beam_searches.clear()
                    _safe_cuda_cleanup()
                    result, rerank_decision = self._recognize_locked(
                        str(input_path),
                        detector_name,
                        1,
                        ctc_weight,
                        mouth_crop_path,
                        rerank_config,
                    )
                    recognition_guidance = (
                        f"GPU memory guard: Beam {beam_size} ran out of memory, so "
                        "recognition completed automatically at Beam 1."
                    )
            text = result.text
            result_state = result.to_dict()
            rerank_status = format_rerank_markdown(rerank_decision)
            if rerank_decision is not None:
                result_state["reranking"] = rerank_decision.to_dict()
            elapsed = time.perf_counter() - started
            audio_path = None
            tts_message = "ℹ️ **Speech:** Auto Read Aloud is off."
            if not text or not text.strip():
                tts_message = (
                    "⚠️ **Speech:** USR 2.0 returned no text, so speech was not generated."
                )
            elif auto_read_aloud:
                try:
                    synthesis = synthesize_speech(
                        text,
                        result_dir,
                        mode=tts_mode,
                        style=expression_level,
                        recognition=result,
                        preserve_timing=bool(preserve_timing),
                        minimax_api_key=minimax_api_key,
                        minimax_region=minimax_region,
                        minimax_api_base=minimax_api_base,
                    )
                    audio_path = synthesis.path
                    tts_message = (
                        f"✅ **Speech:** {synthesis.provider}. {synthesis.note}"
                    )
                except Exception as error:
                    tts_message = self._tts_error(error)
                    LOGGER.exception("Automatic TTS failed: %s", tts_message)
            return (
                str(mouth_crop_path),
                text,
                audio_path,
                format_timeline(result),
                result_state,
                f"{elapsed:.2f} s",
                recognition_guidance,
                tts_message,
                rerank_status,
            )
        except Exception as error:
            message = self._friendly_error(error, detector_name)
            LOGGER.exception("VSR request failed: %s", message)
            if result_dir is not None:
                self.results.discard(result_dir)
            return (
                None,
                "",
                None,
                format_timeline(None),
                None,
                f"{time.perf_counter() - started:.2f} s",
                message,
                "",
                "⚠️ **Language reranking:** Not run because recognition failed.",
            )
        finally:
            _safe_cuda_cleanup()

    def synthesize(
        self,
        result_state: Optional[dict],
        text: str,
        tts_mode: str,
        expression_level: str,
        preserve_timing: bool,
        minimax_region: str = MINIMAX_REGION_AUTO,
        minimax_api_key: str = "",
        minimax_api_base: str = "",
    ) -> Tuple[Optional[str], str]:
        """Gradio callback for manual replay; it never invokes the VSR model."""
        if not text or not text.strip():
            return None, "There is no recognized text to read aloud."

        result_dir = self.results.create()
        try:
            synthesis = synthesize_speech(
                text,
                result_dir,
                mode=tts_mode,
                style=expression_level,
                recognition=result_state,
                preserve_timing=bool(preserve_timing),
                minimax_api_key=minimax_api_key,
                minimax_region=minimax_region,
                minimax_api_base=minimax_api_base,
            )
            return (
                synthesis.path,
                f"✅ **Speech:** {synthesis.provider}. {synthesis.note}",
            )
        except Exception as error:
            self.results.discard(result_dir)
            message = self._tts_error(error)
            LOGGER.exception("Manual TTS failed: %s", message)
            return None, message

    @staticmethod
    def _tts_error(error: Exception) -> str:
        detail = f"{type(error).__name__}: {error}"
        if len(detail) > 520:
            detail = detail[:517] + "..."
        return f"❌ **Speech failed:** {detail}. The recognized text is still available."

    @staticmethod
    def verify_minimax(api_key: str, region: str, api_base: str = "") -> str:
        """Validate a MiniMax key without storing it or generating billable audio."""
        return validate_minimax_key(api_key, region, api_base)

    @staticmethod
    def verify_language_reranker(api_key: str, api_base: str, model: str) -> str:
        """Run one tiny scoring request without storing or logging the key."""
        return test_language_connection(api_key, api_base, model)

    @staticmethod
    def _friendly_error(error: Exception, detector_name: str) -> str:
        text = str(error)
        lower = text.lower()
        if _is_cuda_oom(error):
            return (
                "GPU memory was exhausted. The app already limits this GPU to a safe "
                "Beam size and retries once at Beam 1. CUDA may remain unavailable after "
                "an asynchronous OOM; restart the service once, then use a 2–5 second clip."
            )
        if "could not detect a face" in lower or "not every frame has landmark" in lower:
            return (
                "No stable face/mouth landmarks were found. Use a well-lit clip with "
                "one unobstructed, mostly frontal face."
            )
        if detector_name == "retinaface" and (
            "ibug" in lower or isinstance(error, ImportError)
        ):
            return (
                "RetinaFace/FAN is not installed. Install the optional ibug "
                "face_detection and face_alignment packages, or select MediaPipe."
            )
        if isinstance(error, FileNotFoundError):
            return text
        if isinstance(error, ValueError):
            return text
        return f"Inference failed: {type(error).__name__}: {text}"

    def close(self) -> None:
        for detector in self._detectors.values():
            try:
                detector.close()
            except Exception:
                LOGGER.debug("Detector cleanup failed", exc_info=True)
        self._detectors.clear()


def _settings(prefix: str, safe_beam_limit: int):
    with gr.Accordion(
        "Advanced Settings",
        open=False,
        elem_classes="advanced-accordion",
    ):
        detector = gr.Dropdown(
            choices=["mediapipe", "retinaface"],
            value="mediapipe",
            label="Face detector",
            info="RetinaFace requires the optional ibug packages.",
            key=f"{prefix}_detector",
        )
        beam_size = gr.Slider(
            minimum=1,
            maximum=safe_beam_limit,
            value=safe_beam_limit,
            step=1,
            label="Beam size",
            info=(
                f"Limited to {safe_beam_limit} for the detected GPU to prevent "
                "CUDA memory errors."
            ),
            key=f"{prefix}_beam_size",
        )
        ctc_weight = gr.Slider(
            minimum=0.0,
            maximum=1.0,
            value=0.1,
            step=0.05,
            label="CTC weight",
            info="The project default is 0.1.",
            key=f"{prefix}_ctc_weight",
        )
    return detector, beam_size, ctc_weight


def _outputs(prefix: str):
    gr.Markdown(
        "### Recognition result\nThe transcript remains the source of truth; "
        "speech reconstruction follows its detected timing.",
        elem_classes="section-heading",
    )
    with gr.Row(equal_height=True, elem_classes="evidence-grid"):
        mouth_crop = gr.Video(
            label="Detected mouth region",
            format="mp4",
            interactive=False,
            elem_classes="roi-video",
            key=f"{prefix}_mouth_crop",
        )
        transcription = gr.Textbox(
            label="Recognized Text",
            lines=5,
            interactive=False,
            elem_classes="vsr-result",
            key=f"{prefix}_transcription",
        )
    with gr.Row(elem_classes="playback-actions"):
        auto_read = gr.Checkbox(
            value=True,
            label="Auto Read Aloud",
            info="Generate prosody-aware speech after recognition.",
            key=f"{prefix}_auto_read",
        )
        read_button = gr.Button(
            "Read aloud",
            variant="secondary",
            key=f"{prefix}_read_button",
        )
    speech = gr.Audio(
        label="Reconstructed Speech",
        type="filepath",
        format="mp3",
        autoplay=True,
        interactive=False,
        elem_classes="speech-output",
        key=f"{prefix}_speech",
    )
    tts_error = gr.Markdown(
        "ℹ️ **Speech:** Not generated yet.",
        elem_classes="provider-status",
        key=f"{prefix}_tts_status",
    )
    with gr.Accordion(
        "Voice reconstruction settings",
        open=False,
        elem_classes="settings-accordion",
    ):
        with gr.Row(elem_classes="tts-options"):
            tts_mode = gr.Dropdown(
                choices=TTS_MODE_CHOICES,
                value=MODE_AUTO,
                label="TTS mode",
                info="Automatic prefers MiniMax, then ElevenLabs, then Edge TTS.",
                key=f"{prefix}_tts_mode",
            )
            expression_level = gr.Radio(
                choices=STYLE_LEVEL_CHOICES,
                value=STYLE_NATURAL,
                label="Expression level",
                key=f"{prefix}_expression_level",
            )
        preserve_timing = gr.Checkbox(
            value=True,
            label="Preserve detected video rhythm",
            info="Fit phrase timing and pauses to the original 25 FPS video timeline.",
            key=f"{prefix}_preserve_timing",
        )
    with gr.Accordion(
        "Detected pause timeline",
        open=False,
        elem_classes="timeline-accordion",
    ):
        timeline = gr.Markdown(
            value="*No timing analysis is available yet.*",
            key=f"{prefix}_timeline",
        )
    result_state = gr.State(value=None)
    with gr.Accordion(
        "Diagnostics",
        open=False,
        elem_classes="diagnostics-accordion",
    ):
        with gr.Row():
            elapsed = gr.Textbox(
                label="Inference Time", interactive=False, key=f"{prefix}_elapsed"
            )
            error = gr.Textbox(
                label="Recognition Guidance",
                lines=2,
                interactive=False,
                key=f"{prefix}_error",
            )
        rerank_status = gr.Markdown(
            "ℹ️ **Language reranking:** Off. The visual Top-1 will be used.",
            elem_classes="rerank-result",
            key=f"{prefix}_rerank_status",
        )
    return (
        mouth_crop,
        transcription,
        speech,
        timeline,
        result_state,
        elapsed,
        error,
        tts_error,
        rerank_status,
        auto_read,
        read_button,
        tts_mode,
        expression_level,
        preserve_timing,
    )


def build_app(service: VSRService) -> gr.Blocks:
    """Build the UI around an already initialized singleton service."""
    theme = gr.themes.Base(
        primary_hue=gr.themes.colors.orange,
        neutral_hue=gr.themes.colors.slate,
        font=["Inter", "ui-sans-serif", "system-ui", "sans-serif"],
    ).set(
        body_background_fill="oklch(0.975 0.004 250)",
        body_background_fill_dark="oklch(0.145 0.012 250)",
        body_text_color="oklch(0.22 0.025 250)",
        body_text_color_dark="oklch(0.93 0.008 250)",
        body_text_color_subdued="oklch(0.41 0.025 250)",
        body_text_color_subdued_dark="oklch(0.72 0.018 250)",
        background_fill_primary="oklch(1 0 0)",
        background_fill_primary_dark="oklch(0.17 0.014 250)",
        background_fill_secondary="oklch(0.94 0.008 250)",
        background_fill_secondary_dark="oklch(0.205 0.016 250)",
        border_color_primary="oklch(0.86 0.012 250)",
        border_color_primary_dark="oklch(0.34 0.018 250)",
        block_background_fill="oklch(1 0 0)",
        block_background_fill_dark="oklch(0.19 0.015 250)",
        block_border_color="oklch(0.86 0.012 250)",
        block_border_color_dark="oklch(0.34 0.018 250)",
        block_info_text_color="oklch(0.41 0.025 250)",
        block_info_text_color_dark="oklch(0.72 0.018 250)",
        block_label_text_color="oklch(0.31 0.025 250)",
        block_label_text_color_dark="oklch(0.86 0.012 250)",
        block_shadow="none",
        block_shadow_dark="none",
        panel_background_fill="oklch(0.94 0.008 250)",
        panel_background_fill_dark="oklch(0.205 0.016 250)",
        input_background_fill="oklch(1 0 0)",
        input_background_fill_dark="oklch(0.225 0.016 250)",
        input_background_fill_focus="oklch(1 0 0)",
        input_background_fill_focus_dark="oklch(0.245 0.018 250)",
        input_border_color="oklch(0.80 0.016 250)",
        input_border_color_dark="oklch(0.39 0.022 250)",
        input_border_color_focus="oklch(0.55 0.16 68)",
        input_border_color_focus_dark="oklch(0.68 0.14 68)",
        input_placeholder_color="oklch(0.43 0.025 250)",
        input_placeholder_color_dark="oklch(0.70 0.018 250)",
        accordion_text_color="oklch(0.25 0.025 250)",
        accordion_text_color_dark="oklch(0.90 0.010 250)",
        button_primary_background_fill="oklch(0.55 0.16 68)",
        button_primary_background_fill_dark="oklch(0.59 0.16 68)",
        button_primary_background_fill_hover="oklch(0.49 0.15 68)",
        button_primary_background_fill_hover_dark="oklch(0.65 0.15 68)",
        button_primary_text_color="oklch(1 0 0)",
        button_primary_text_color_dark="oklch(1 0 0)",
        button_primary_shadow="none",
        button_primary_shadow_dark="none",
        button_secondary_background_fill="oklch(0.94 0.008 250)",
        button_secondary_background_fill_dark="oklch(0.27 0.020 250)",
        button_secondary_text_color="oklch(0.30 0.070 235)",
        button_secondary_text_color_dark="oklch(0.88 0.030 235)",
        button_secondary_shadow="none",
        button_secondary_shadow_dark="none",
    )
    css = """
    .gradio-container {
      --vsr-primary: oklch(0.55 0.16 68);
      --vsr-primary-hover: oklch(0.49 0.15 68);
      --vsr-accent: oklch(0.42 0.09 235);
      --vsr-focus: oklch(0.55 0.16 68 / 0.20);
      width: 100% !important;
      max-width: none !important;
      padding: clamp(10px, 2vw, 28px) !important;
      background: var(--body-background-fill) !important;
      color: var(--body-text-color) !important;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont,
        "Segoe UI", sans-serif !important;
    }
    main.contain {
      width: 100% !important;
      max-width: none !important;
      margin: 0 !important;
    }
    .gradio-container .main.fillable {
      width: 100% !important;
      max-width: none !important;
      margin: 0 !important;
      padding: 0 !important;
    }
    .app-toolbar {
      align-items: center !important;
      gap: 20px !important;
      padding: 12px 16px;
      border: 1px solid var(--border-color-primary);
      border-radius: 12px;
      background: var(--block-background-fill);
    }
    .brand-column { gap: 4px !important; min-width: 420px; }
    .vsr-brand { display: flex; align-items: center; gap: 12px; min-width: 0; }
    .vsr-mark {
      display: grid;
      place-items: center;
      width: 40px;
      height: 40px;
      flex: 0 0 40px;
      border-radius: 8px;
      background: var(--vsr-primary);
      color: white;
      font-size: 0.72rem;
      font-weight: 800;
      letter-spacing: 0.04em;
    }
    .vsr-identity h1 {
      margin: 0;
      color: var(--body-text-color);
      font-size: 1.18rem;
      font-weight: 720;
      line-height: 1.2;
      letter-spacing: -0.02em;
      text-wrap: balance;
    }
    .vsr-identity p {
      margin: 2px 0 0;
      color: var(--body-text-color-subdued);
      font-size: 0.82rem;
      line-height: 1.35;
    }
    .model-status { min-width: 260px; }
    .model-status .prose, .model-status p {
      margin: 0 !important;
      color: var(--body-text-color-subdued) !important;
      font-size: 0.78rem !important;
    }
    .model-status strong { color: var(--body-text-color) !important; }
    .model-status code { white-space: normal !important; overflow-wrap: anywhere; }
    .api-connect-panel {
      gap: 6px !important;
      padding: 10px 12px;
      border: 1px solid var(--border-color-primary);
      border-radius: 10px;
      background: color-mix(in srgb, var(--block-background-fill) 84%, var(--primary-500) 16%);
    }
    .api-connect-title .prose, .api-connect-title p {
      margin: 0 !important;
      color: var(--body-text-color) !important;
      font-size: 0.8rem !important;
      font-weight: 700 !important;
      letter-spacing: 0.02em;
    }
    .api-key-field { min-width: 280px; }
    .api-url-field { min-width: 360px; }
    .api-key-field label > span {
      color: var(--body-text-color) !important;
      font-weight: 650;
    }
    .api-connect-row { align-items: flex-end !important; gap: 8px !important; }
    .api-connect-row button { min-width: 112px; }
    .key-status .prose, .key-status p {
      margin: 0 !important;
      color: var(--body-text-color-subdued) !important;
      font-size: 0.78rem !important;
      line-height: 1.4 !important;
    }
    .reranker-panel {
      margin-top: 10px;
      overflow: hidden !important;
      border-color: var(--border-color-primary) !important;
      border-radius: 12px !important;
      background: var(--block-background-fill) !important;
    }
    .reranker-intro p {
      margin: 0 !important;
      color: var(--body-text-color-subdued) !important;
      font-size: 0.84rem !important;
    }
    .reranker-options { align-items: flex-end !important; gap: 10px !important; }
    .reranker-toggle {
      width: 100% !important;
      max-width: none !important;
      padding: 8px 10px;
      border-radius: 8px;
      background: var(--background-fill-secondary);
    }
    .rerank-result {
      margin-top: 8px;
      padding: 8px 10px;
      border: 1px solid var(--border-color-primary);
      border-radius: 8px;
      background: var(--background-fill-secondary);
      overflow-x: auto;
    }
    .rerank-result .prose, .rerank-result p {
      margin: 0 !important;
      font-size: 0.8rem !important;
      line-height: 1.45 !important;
    }
    .workspace-tabs { margin-top: 10px; background: transparent; }
    .workspace-tabs .tab-nav {
      gap: 4px;
      border-bottom: 1px solid var(--border-color-primary);
    }
    .workspace-tabs .tab-nav button {
      min-height: 44px;
      padding-inline: 16px;
      color: var(--body-text-color-subdued);
      font-weight: 600;
    }
    .workspace-tabs .tab-nav button.selected {
      color: var(--vsr-primary-hover);
      border-color: var(--vsr-primary);
    }
    .tab-intro { margin: 8px 0 0; color: var(--body-text-color-subdued); }
    .tab-intro p { margin: 0 !important; }
    .workspace-grid {
      display: grid !important;
      grid-template-columns: minmax(360px, 0.78fr) minmax(560px, 1.42fr);
      align-items: flex-start !important;
      gap: 16px !important;
      margin-top: 8px;
    }
    .input-rail, .result-rail {
      gap: 12px !important;
      padding: 16px;
      border-radius: 12px;
    }
    .input-rail {
      width: 100% !important;
      max-width: none;
      background: var(--background-fill-secondary);
    }
    .result-rail {
      width: 100% !important;
      min-width: 0 !important;
      border: 1px solid var(--border-color-primary);
      background: var(--block-background-fill);
    }
    .section-heading { max-width: 72ch; }
    .section-heading h3 {
      margin: 0 0 4px !important;
      color: var(--body-text-color);
      font-size: 1rem !important;
      font-weight: 680;
      line-height: 1.35;
    }
    .section-heading p {
      margin: 0 !important;
      color: var(--body-text-color-subdued);
      font-size: 0.86rem;
      line-height: 1.45;
    }
    .input-video, .roi-video {
      background: var(--input-background-fill);
      border-radius: 10px;
    }
    .input-video { min-height: clamp(240px, 30vh, 360px); }
    .evidence-grid { align-items: stretch !important; gap: 12px !important; }
    .evidence-grid > * { min-width: 280px; }
    .roi-video { min-height: 218px; }
    .vsr-result textarea {
      min-height: 178px !important;
      color: var(--body-text-color) !important;
      font-size: 1.12rem !important;
      font-weight: 650;
      line-height: 1.55 !important;
    }
    .playback-actions { align-items: stretch !important; gap: 12px !important; }
    .playback-actions > * { min-height: 56px; }
    .speech-output { min-height: 104px; }
    .provider-status {
      padding: 9px 12px;
      border: 1px solid var(--border-color-primary);
      border-radius: 9px;
      background: var(--background-fill-secondary);
    }
    .provider-status .prose, .provider-status p {
      margin: 0 !important;
      color: var(--body-text-color) !important;
      font-size: 0.82rem !important;
      line-height: 1.45 !important;
      overflow-wrap: anywhere;
    }
    .advanced-accordion, .settings-accordion,
    .timeline-accordion, .diagnostics-accordion {
      overflow-x: hidden !important;
      border-color: var(--border-color-primary) !important;
      border-radius: 10px !important;
      background: var(--block-background-fill) !important;
    }
    .advanced-accordion button.label-wrap,
    .settings-accordion button.label-wrap,
    .timeline-accordion button.label-wrap,
    .diagnostics-accordion button.label-wrap {
      width: 100% !important;
      max-width: 100% !important;
    }
    .tts-options { align-items: flex-start !important; gap: 12px !important; }
    .tts-options > * { min-width: 280px; }
    button { min-height: 44px; transition: 180ms cubic-bezier(0.22, 1, 0.36, 1); }
    button.primary {
      border-color: transparent !important;
      box-shadow: none !important;
      font-weight: 700 !important;
    }
    button.secondary { font-weight: 650 !important; }
    button:focus-visible, input:focus-visible, textarea:focus-visible {
      outline: 2px solid var(--vsr-primary) !important;
      outline-offset: 2px !important;
      box-shadow: 0 0 0 3px var(--vsr-focus) !important;
    }
    input, textarea, select { color: var(--body-text-color) !important; }
    input::placeholder, textarea::placeholder {
      color: var(--input-placeholder-color) !important;
      opacity: 1 !important;
    }
    .info-text, .secondary-text { color: var(--body-text-color-subdued) !important; }
    @media (max-width: 1120px) {
      .app-toolbar { flex-direction: column !important; }
      .app-toolbar > * { width: 100% !important; min-width: 0 !important; }
      .workspace-grid { grid-template-columns: minmax(0, 1fr); }
      .workspace-grid > * {
        width: 100% !important;
        max-width: none !important;
        min-width: 0 !important;
      }
    }
    @media (max-width: 760px) {
      .api-connect-row, .evidence-grid, .tts-options, .reranker-options {
        flex-direction: column !important;
      }
      .api-connect-row > *, .evidence-grid > *, .tts-options > *, .reranker-options > * {
        width: 100% !important;
        min-width: 0 !important;
      }
      .brand-column, .api-key-field { min-width: 0; }
      .input-rail, .result-rail { padding: 12px; }
      .playback-actions { flex-direction: column !important; }
    }
    @media (prefers-reduced-motion: reduce) {
      *, *::before, *::after {
        scroll-behavior: auto !important;
        transition-duration: 0.01ms !important;
        animation-duration: 0.01ms !important;
        animation-iteration-count: 1 !important;
      }
    }
    """
    with gr.Blocks(
        title="USR 2.0 Visual Speech Recognition",
        delete_cache=(3600, 3600),
        theme=theme,
        css=css,
    ) as app:
        with gr.Row(equal_height=True, elem_classes="app-toolbar"):
            with gr.Column(scale=7, min_width=420, elem_classes="brand-column"):
                gr.HTML(
                    """
                    <div class="vsr-brand">
                      <div class="vsr-mark" aria-hidden="true">USR</div>
                      <div class="vsr-identity">
                        <h1>Visual Speech Studio</h1>
                        <p>Visual lip reading with pause-aware speech reconstruction.</p>
                      </div>
                    </div>
                    """
                )
                gr.Markdown(
                    service.status_message,
                    elem_classes="model-status",
                    container=False,
                )
            with gr.Column(
                scale=5,
                min_width=620,
                elem_classes="api-connect-panel",
            ):
                gr.Markdown(
                    "MiniMax Speech connection · official and RelayRouter routes",
                    elem_classes="api-connect-title",
                    container=False,
                )
                with gr.Row(elem_classes="api-connect-row"):
                    minimax_api_base = gr.Textbox(
                        label="API Base URL (optional)",
                        placeholder="https://provider.example/v1",
                        info="RelayRouter /v1 is mapped automatically to /minimax/v1.",
                        elem_classes="api-url-field",
                        scale=4,
                        min_width=360,
                    )
                    minimax_region = gr.Dropdown(
                        choices=MINIMAX_REGION_CHOICES,
                        value=MINIMAX_REGION_AUTO,
                        label="Official region",
                        info="Used only when Base URL is blank.",
                        scale=2,
                        min_width=180,
                    )
                with gr.Row(elem_classes="api-connect-row"):
                    minimax_api_key = gr.Textbox(
                        label="TTS API key (optional)",
                        type="password",
                        placeholder="Session-only · not saved",
                        info="Enter a new key, then verify it.",
                        elem_classes="api-key-field",
                        scale=3,
                        min_width=280,
                    )
                    verify_minimax_button = gr.Button(
                        "Test connection",
                        variant="secondary",
                        scale=1,
                    )
                minimax_key_status = gr.Markdown(
                    "Connection not tested · RelayRouter checks /v1/models without generating audio.",
                    elem_classes="key-status",
                )

        with gr.Accordion(
            "Language reranking · accuracy-first Top-5 selection",
            open=False,
            elem_classes="reranker-panel",
        ):
            gr.Markdown(
                "Qwen directly selects one candidate from the current video's live Top-5. "
                "It cannot rewrite or generate text outside those candidates.",
                elem_classes="reranker-intro",
            )
            rerank_enabled = gr.Checkbox(
                value=False,
                label="Enable language reranking",
                info="Accuracy first. Requires Beam 2+; API failure keeps visual Top-1.",
                elem_classes="reranker-toggle",
            )
            with gr.Row(elem_classes="reranker-options"):
                rerank_api_base = gr.Textbox(
                    value=DEFAULT_RERANK_API_BASE,
                    label="Language API Base URL",
                    info="OpenAI-compatible endpoint; /chat/completions is added automatically.",
                    scale=5,
                    min_width=360,
                )
                rerank_model = gr.Textbox(
                    value=DEFAULT_RERANK_MODEL,
                    label="Text model",
                    info="A fast non-thinking model is sufficient.",
                    scale=2,
                    min_width=190,
                )
            with gr.Row(elem_classes="reranker-options"):
                rerank_api_key = gr.Textbox(
                    label="Language API key",
                    type="password",
                    placeholder="Session-only · not saved or logged",
                    scale=4,
                    min_width=360,
                )
                verify_rerank_button = gr.Button(
                    "Test language API",
                    variant="secondary",
                    scale=1,
                    min_width=170,
                )
            rerank_key_status = gr.Markdown(
                "Connection not tested · the test sends one very small billable request.",
                elem_classes="key-status",
            )

        verify_minimax_button.click(
            service.verify_minimax,
            inputs=[minimax_api_key, minimax_region, minimax_api_base],
            outputs=[minimax_key_status],
            concurrency_limit=2,
            concurrency_id=TTS_CONCURRENCY_ID,
            api_name=False,
        )
        verify_rerank_button.click(
            service.verify_language_reranker,
            inputs=[rerank_api_key, rerank_api_base, rerank_model],
            outputs=[rerank_key_status],
            concurrency_limit=2,
            concurrency_id="usr2_language_reranking",
            api_name=False,
        )

        with gr.Tabs(elem_classes="workspace-tabs"):
            with gr.Tab("Upload clip"):
                gr.Markdown(
                    "Upload a short browser-supported video. A clear, frontal face "
                    "and a 2–5 second clip produce the most reliable result.",
                    elem_classes="tab-intro",
                )
                with gr.Row(equal_height=False, elem_classes="workspace-grid"):
                    with gr.Column(
                        scale=10,
                        min_width=360,
                        elem_classes="input-rail",
                    ):
                        gr.Markdown(
                            "### Input video\nProvide the visual signal to analyze.",
                            elem_classes="section-heading",
                        )
                        upload_video = gr.Video(
                            sources=["upload"],
                            format="mp4",
                            label="Original video",
                            include_audio=True,
                            elem_classes="input-video",
                        )
                        upload_button = gr.Button(
                            "Recognize lip movements",
                            variant="primary",
                            elem_classes="recognize-button",
                        )
                        upload_detector, upload_beam, upload_ctc = _settings(
                            "upload", service.safe_beam_limit
                        )
                    with gr.Column(
                        scale=12,
                        min_width=420,
                        elem_classes="result-rail",
                    ):
                        (
                            upload_roi,
                            upload_text,
                            upload_speech,
                            upload_timeline,
                            upload_result_state,
                            upload_time,
                            upload_error,
                            upload_tts_error,
                            upload_rerank_status,
                            upload_auto_read,
                            upload_read_button,
                            upload_tts_mode,
                            upload_expression,
                            upload_preserve_timing,
                        ) = _outputs("upload")
                upload_button.click(
                    service.recognize,
                    inputs=[
                        upload_video,
                        upload_detector,
                        upload_beam,
                        upload_ctc,
                        upload_auto_read,
                        upload_tts_mode,
                        upload_expression,
                        upload_preserve_timing,
                        minimax_region,
                        minimax_api_key,
                        minimax_api_base,
                        rerank_enabled,
                        rerank_api_base,
                        rerank_model,
                        rerank_api_key,
                    ],
                    outputs=[
                        upload_roi,
                        upload_text,
                        upload_speech,
                        upload_timeline,
                        upload_result_state,
                        upload_time,
                        upload_error,
                        upload_tts_error,
                        upload_rerank_status,
                    ],
                    concurrency_limit=1,
                    concurrency_id=GPU_CONCURRENCY_ID,
                    api_name="recognize_upload",
                )
                upload_read_button.click(
                    service.synthesize,
                    inputs=[
                        upload_result_state,
                        upload_text,
                        upload_tts_mode,
                        upload_expression,
                        upload_preserve_timing,
                        minimax_region,
                        minimax_api_key,
                        minimax_api_base,
                    ],
                    outputs=[upload_speech, upload_tts_error],
                    concurrency_limit=2,
                    concurrency_id=TTS_CONCURRENCY_ID,
                    api_name="read_upload_result",
                )

            with gr.Tab("Camera clip"):
                gr.Markdown(
                    "Record for 2–5 seconds, stop, then recognize the complete clip. "
                    "USR 2.0 analyzes sequences rather than frame-by-frame streaming.",
                    elem_classes="tab-intro",
                )
                with gr.Row(equal_height=False, elem_classes="workspace-grid"):
                    with gr.Column(
                        scale=10,
                        min_width=360,
                        elem_classes="input-rail",
                    ):
                        gr.Markdown(
                            "### Camera input\nRecord one clear, mostly frontal speaker.",
                            elem_classes="section-heading",
                        )
                        camera_video = gr.Video(
                            sources=["webcam"],
                            format="mp4",
                            label="Camera recording",
                            include_audio=False,
                            elem_classes="input-video",
                        )
                        camera_button = gr.Button(
                            "Recognize recorded clip",
                            variant="primary",
                            elem_classes="recognize-button",
                        )
                        camera_detector, camera_beam, camera_ctc = _settings(
                            "camera", service.safe_beam_limit
                        )
                    with gr.Column(
                        scale=12,
                        min_width=420,
                        elem_classes="result-rail",
                    ):
                        (
                            camera_roi,
                            camera_text,
                            camera_speech,
                            camera_timeline,
                            camera_result_state,
                            camera_time,
                            camera_error,
                            camera_tts_error,
                            camera_rerank_status,
                            camera_auto_read,
                            camera_read_button,
                            camera_tts_mode,
                            camera_expression,
                            camera_preserve_timing,
                        ) = _outputs("camera")
                camera_button.click(
                    service.recognize,
                    inputs=[
                        camera_video,
                        camera_detector,
                        camera_beam,
                        camera_ctc,
                        camera_auto_read,
                        camera_tts_mode,
                        camera_expression,
                        camera_preserve_timing,
                        minimax_region,
                        minimax_api_key,
                        minimax_api_base,
                        rerank_enabled,
                        rerank_api_base,
                        rerank_model,
                        rerank_api_key,
                    ],
                    outputs=[
                        camera_roi,
                        camera_text,
                        camera_speech,
                        camera_timeline,
                        camera_result_state,
                        camera_time,
                        camera_error,
                        camera_tts_error,
                        camera_rerank_status,
                    ],
                    concurrency_limit=1,
                    concurrency_id=GPU_CONCURRENCY_ID,
                    api_name="recognize_camera",
                )
                camera_read_button.click(
                    service.synthesize,
                    inputs=[
                        camera_result_state,
                        camera_text,
                        camera_tts_mode,
                        camera_expression,
                        camera_preserve_timing,
                        minimax_region,
                        minimax_api_key,
                        minimax_api_base,
                    ],
                    outputs=[camera_speech, camera_tts_error],
                    concurrency_limit=2,
                    concurrency_id=TTS_CONCURRENCY_ID,
                    api_name="read_camera_result",
                )

    return app.queue(max_size=8, default_concurrency_limit=1, api_open=False)


def main() -> None:
    logging.basicConfig(
        level=os.environ.get("USR2_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    service = VSRService(
        Path(os.environ.get("USR2_CHECKPOINT", str(DEFAULT_CHECKPOINT)))
    )
    app = build_app(service)
    app.launch(
        server_name=os.environ.get("USR2_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.environ.get("USR2_SERVER_PORT", "7860")),
        share=os.environ.get("USR2_SHARE", "0").lower() in {"1", "true", "yes"},
        show_error=True,
        max_file_size="500mb",
        allowed_paths=[str(service.results.root)],
    )


if __name__ == "__main__":
    main()
