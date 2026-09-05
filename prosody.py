"""Timing and visual-prosody helpers for lip-reading results.

The recognizer remains responsible for *what* was said.  This module keeps the
CTC timing evidence that was previously discarded and turns only sufficiently
supported inter-word gaps into pause events.  It never invents spoken filler
words such as "uh" or "um".
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np


DEFAULT_MIN_PAUSE_S = 0.12
DEFAULT_MAX_PAUSE_ACTIVITY = 0.60
DEFAULT_MIN_BLANK_CONFIDENCE = 0.42


@dataclass
class TokenSpan:
    token_id: int
    token: str
    start_frame: int
    end_frame: int
    confidence: float

    @property
    def duration_frames(self) -> int:
        return self.end_frame - self.start_frame + 1


@dataclass
class WordSpan:
    text: str
    start_frame: int
    end_frame: int
    confidence: float
    token_start: int
    token_end: int


@dataclass
class PauseEvent:
    after_word: int
    before_word: int
    start_frame: int
    end_frame: int
    start_s: float
    end_s: float
    duration_s: float
    kind: str
    confidence: float
    blank_confidence: float
    inactivity: float


@dataclass
class RecognitionResult:
    text: str
    token_ids: list[int] = field(default_factory=list)
    tokens: list[str] = field(default_factory=list)
    token_spans: list[TokenSpan] = field(default_factory=list)
    words: list[WordSpan] = field(default_factory=list)
    pauses: list[PauseEvent] = field(default_factory=list)
    frame_duration_s: float = 0.04
    video_duration_s: float = 0.0
    alignment_status: str = "not-run"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "RecognitionResult":
        return cls(
            text=value.get("text", ""),
            token_ids=list(value.get("token_ids", [])),
            tokens=list(value.get("tokens", [])),
            token_spans=[TokenSpan(**item) for item in value.get("token_spans", [])],
            words=[WordSpan(**item) for item in value.get("words", [])],
            pauses=[PauseEvent(**item) for item in value.get("pauses", [])],
            frame_duration_s=float(value.get("frame_duration_s", 0.04)),
            video_duration_s=float(value.get("video_duration_s", 0.0)),
            alignment_status=value.get("alignment_status", "unknown"),
        )


def _as_numpy(log_probs: Any) -> np.ndarray:
    if hasattr(log_probs, "detach"):
        log_probs = log_probs.detach().float().cpu().numpy()
    array = np.asarray(log_probs, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"Expected CTC log probabilities shaped (T, V), got {array.shape}")
    return array


def ctc_forced_align(
    log_probs: Any,
    token_ids: Sequence[int],
    token_list: Sequence[str],
    blank_id: int = 0,
) -> list[TokenSpan]:
    """Viterbi-align one decoded CTC token sequence to encoder frames.

    ``log_probs`` must already be log-softmax probabilities.  Repeated target
    tokens follow the standard CTC rule and therefore cannot use a two-state
    skip transition.
    """
    scores = _as_numpy(log_probs)
    target = [int(token_id) for token_id in token_ids]
    if not target:
        return []
    if any(token_id <= blank_id or token_id >= scores.shape[1] for token_id in target):
        raise ValueError("The decoded token sequence contains an invalid CTC label")

    frames = scores.shape[0]
    states = 2 * len(target) + 1
    if frames < len(target):
        raise ValueError(
            f"CTC alignment needs at least {len(target)} frames, but received {frames}"
        )

    labels = np.full(states, blank_id, dtype=np.int64)
    labels[1::2] = target
    neg_inf = -np.inf
    dp = np.full((frames, states), neg_inf, dtype=np.float64)
    back = np.full((frames, states), -1, dtype=np.int32)

    dp[0, 0] = scores[0, blank_id]
    if states > 1:
        dp[0, 1] = scores[0, labels[1]]

    for frame in range(1, frames):
        max_state = min(states, 2 * frame + 2)
        for state in range(max_state):
            candidates = [(dp[frame - 1, state], state)]
            if state > 0:
                candidates.append((dp[frame - 1, state - 1], state - 1))
            if (
                state > 1
                and labels[state] != blank_id
                and labels[state] != labels[state - 2]
            ):
                candidates.append((dp[frame - 1, state - 2], state - 2))
            best_score, best_previous = max(candidates, key=lambda item: item[0])
            if np.isfinite(best_score):
                dp[frame, state] = best_score + scores[frame, labels[state]]
                back[frame, state] = best_previous

    final_candidates = [states - 1]
    if states > 1:
        final_candidates.append(states - 2)
    final_state = max(final_candidates, key=lambda state: dp[-1, state])
    if not np.isfinite(dp[-1, final_state]):
        raise ValueError("The decoded token sequence could not be aligned to the video")

    path = np.empty(frames, dtype=np.int32)
    state = final_state
    for frame in range(frames - 1, -1, -1):
        path[frame] = state
        if frame > 0:
            state = back[frame, state]
            if state < 0:
                raise ValueError("CTC backtracking reached an invalid state")

    spans: list[TokenSpan] = []
    for target_index, token_id in enumerate(target):
        token_state = 2 * target_index + 1
        token_frames = np.flatnonzero(path == token_state)
        if token_frames.size == 0:
            raise ValueError("CTC alignment skipped a decoded token")
        token_scores = scores[token_frames, token_id]
        confidence = float(np.exp(np.clip(token_scores.mean(), -30.0, 0.0)))
        spans.append(
            TokenSpan(
                token_id=token_id,
                token=token_list[token_id],
                start_frame=int(token_frames[0]),
                end_frame=int(token_frames[-1]),
                confidence=confidence,
            )
        )
    return spans


def merge_token_spans_into_words(token_spans: Sequence[TokenSpan]) -> list[WordSpan]:
    """Merge SentencePiece-style ``▁`` units into word spans."""
    words: list[WordSpan] = []
    current_text = ""
    current_spans: list[TokenSpan] = []
    current_start_index = 0

    def flush() -> None:
        nonlocal current_text, current_spans, current_start_index
        cleaned = current_text.strip()
        if cleaned and current_spans:
            weights = np.array([span.duration_frames for span in current_spans], dtype=float)
            confidences = np.array([span.confidence for span in current_spans], dtype=float)
            words.append(
                WordSpan(
                    text=cleaned,
                    start_frame=current_spans[0].start_frame,
                    end_frame=current_spans[-1].end_frame,
                    confidence=float(np.average(confidences, weights=weights)),
                    token_start=current_start_index,
                    token_end=current_start_index + len(current_spans) - 1,
                )
            )
        current_text = ""
        current_spans = []

    for token_index, span in enumerate(token_spans):
        piece = span.token.replace("<eos>", "")
        starts_word = piece.startswith("▁")
        if starts_word and current_spans:
            flush()
        if not current_spans:
            current_start_index = token_index
        piece = piece[1:] if starts_word else piece
        if piece == "<unk>":
            piece = "?"
        current_text += piece
        current_spans.append(span)
    flush()
    return words


def compute_mouth_activity(mouth_video: Any, target_frames: int | None = None) -> np.ndarray:
    """Return a robust, normalized per-frame mouth-motion curve in ``[0, 1]``."""
    video = np.asarray(mouth_video)
    if video.ndim not in (3, 4) or video.shape[0] == 0:
        length = target_frames or 0
        return np.zeros(length, dtype=np.float32)
    gray = video.astype(np.float32)
    if gray.ndim == 4:
        gray = gray.mean(axis=-1)
    difference = np.mean(np.abs(np.diff(gray, axis=0)), axis=(1, 2))
    activity = np.concatenate(([0.0], difference))
    if activity.size >= 3:
        activity = np.convolve(activity, np.ones(3) / 3.0, mode="same")
    scale = float(np.percentile(activity, 90))
    if scale > 1e-6:
        activity = np.clip(activity / scale, 0.0, 1.0)
    else:
        activity = np.zeros_like(activity)

    if target_frames is not None and target_frames > 0 and len(activity) != target_frames:
        source_axis = np.linspace(0.0, 1.0, len(activity))
        target_axis = np.linspace(0.0, 1.0, target_frames)
        activity = np.interp(target_axis, source_axis, activity)
    return activity.astype(np.float32)


def detect_pauses(
    words: Sequence[WordSpan],
    log_probs: Any,
    mouth_activity: Sequence[float],
    frame_duration_s: float,
    *,
    blank_id: int = 0,
    min_pause_s: float = DEFAULT_MIN_PAUSE_S,
    max_activity: float = DEFAULT_MAX_PAUSE_ACTIVITY,
    min_blank_confidence: float = DEFAULT_MIN_BLANK_CONFIDENCE,
) -> list[PauseEvent]:
    """Detect inter-word pauses from contiguous CTC/visual stillness evidence.

    Forced alignment can assign one or two true pause frames to either adjacent
    word.  Short gaps therefore get a small, stricter recovery window around the
    boundary.  Longer gaps are segmented so lip motion at their edges no longer
    causes an otherwise clear still interval to be discarded.
    """
    scores = _as_numpy(log_probs)
    activity = np.asarray(mouth_activity, dtype=np.float64)
    if len(activity) != scores.shape[0]:
        raise ValueError("Mouth activity and CTC probabilities must share a time axis")
    if frame_duration_s <= 0:
        raise ValueError("Frame duration must be positive")

    pauses: list[PauseEvent] = []
    minimum_frames = max(1, int(np.ceil(min_pause_s / frame_duration_s)))
    blank_probabilities = np.exp(np.clip(scores[:, blank_id], -30.0, 0.0))

    def candidate_runs(start: int, end: int, *, recovery: bool) -> list[tuple[int, int]]:
        if end < start:
            return []
        gap_activity = activity[start : end + 1]
        gap_blank = blank_probabilities[start : end + 1]
        if recovery:
            # Frames borrowed from aligned word edges need substantially stronger
            # agreement before they may be treated as silence.
            mask = (gap_activity <= min(max_activity, 0.44)) & (
                gap_blank >= max(min_blank_confidence, 0.62)
            )
        else:
            evidence = 0.54 * gap_blank + 0.46 * np.clip(
                1.0 - gap_activity, 0.0, 1.0
            )
            mask = (
                (gap_activity <= max_activity)
                & (gap_blank >= min_blank_confidence)
                & (evidence >= 0.54)
            )

        runs: list[tuple[int, int]] = []
        run_start: int | None = None
        for offset, is_pause_frame in enumerate(mask):
            if is_pause_frame and run_start is None:
                run_start = offset
            if run_start is not None and (
                not is_pause_frame or offset == len(mask) - 1
            ):
                run_end = offset if is_pause_frame else offset - 1
                if run_end - run_start + 1 >= minimum_frames:
                    runs.append((start + run_start, start + run_end))
                run_start = None
        return runs

    for word_index in range(len(words) - 1):
        raw_start = words[word_index].end_frame + 1
        raw_end = words[word_index + 1].start_frame - 1
        raw_frame_count = raw_end - raw_start + 1
        recovery = raw_frame_count < minimum_frames
        if recovery:
            padding = max(1, int(np.ceil(0.12 / frame_duration_s)))
            boundary = (words[word_index].end_frame + words[word_index + 1].start_frame) // 2
            search_start = max(0, boundary - padding)
            search_end = min(len(activity) - 1, boundary + padding)
        else:
            search_start = raw_start
            search_end = raw_end

        runs = candidate_runs(search_start, search_end, recovery=recovery)
        if not runs:
            continue

        # Prefer the strongest sustained interval, then duration. This avoids
        # selecting a long but marginally still run over a clear closed-mouth run.
        def run_score(run: tuple[int, int]) -> tuple[float, int]:
            run_start, run_end = run
            mean_blank = float(blank_probabilities[run_start : run_end + 1].mean())
            mean_inactivity = float(
                1.0 - activity[run_start : run_end + 1].mean()
            )
            return (0.54 * mean_blank + 0.46 * mean_inactivity, run_end - run_start)

        start_frame, end_frame = max(runs, key=run_score)
        frame_count = end_frame - start_frame + 1
        gap_activity = float(activity[start_frame : end_frame + 1].mean())
        inactivity = float(np.clip(1.0 - gap_activity, 0.0, 1.0))
        blank_confidence = float(
            blank_probabilities[start_frame : end_frame + 1].mean()
        )

        duration_s = frame_count * frame_duration_s
        confidence = float(
            np.clip(
                (0.54 * blank_confidence + 0.46 * inactivity)
                * (0.94 if recovery else 1.0),
                0.0,
                1.0,
            )
        )
        if duration_s >= 0.68:
            kind = "long"
        elif duration_s >= 0.32 and confidence >= 0.68:
            kind = "breath"
        else:
            kind = "short"
        pauses.append(
            PauseEvent(
                after_word=word_index,
                before_word=word_index + 1,
                start_frame=start_frame,
                end_frame=end_frame,
                start_s=round(start_frame * frame_duration_s, 4),
                end_s=round((end_frame + 1) * frame_duration_s, 4),
                duration_s=round(duration_s, 4),
                kind=kind,
                confidence=round(confidence, 4),
                blank_confidence=round(blank_confidence, 4),
                inactivity=round(inactivity, 4),
            )
        )
    return pauses


def build_recognition_result(
    text: str,
    token_ids: Sequence[int],
    token_list: Sequence[str],
    log_probs: Any,
    mouth_video: Any,
    video_duration_s: float,
) -> RecognitionResult:
    """Create an aligned result while degrading safely when alignment fails."""
    scores = _as_numpy(log_probs)
    frame_duration_s = (
        float(video_duration_s) / scores.shape[0] if scores.shape[0] else 0.04
    )
    result = RecognitionResult(
        text=text,
        token_ids=[int(token_id) for token_id in token_ids],
        tokens=[token_list[int(token_id)] for token_id in token_ids],
        frame_duration_s=frame_duration_s,
        video_duration_s=float(video_duration_s),
    )
    try:
        result.token_spans = ctc_forced_align(scores, token_ids, token_list)
        result.words = merge_token_spans_into_words(result.token_spans)
        activity = compute_mouth_activity(mouth_video, target_frames=scores.shape[0])
        result.pauses = detect_pauses(
            result.words,
            scores,
            activity,
            frame_duration_s,
        )
        result.alignment_status = "ok"
    except (ValueError, IndexError) as error:
        result.alignment_status = f"unavailable: {error}"
    return result


def format_timeline(result: RecognitionResult | dict[str, Any] | None) -> str:
    if not result:
        return "*No timing analysis is available yet.*"
    if isinstance(result, dict):
        result = RecognitionResult.from_dict(result)
    if result.alignment_status != "ok":
        return f"*Timing analysis {result.alignment_status}.*"
    if not result.pauses:
        boundary_count = max(0, len(result.words) - 1)
        if boundary_count == 0:
            return (
                "**Timing analysis:** the aligned result has fewer than two words, "
                "so there is no inter-word boundary to evaluate."
            )
        return (
            f"**Timing analysis:** analyzed {boundary_count} word boundaries; no "
            f"pause met the {DEFAULT_MIN_PAUSE_S:.2f}s visual/CTC evidence threshold."
        )

    lines = [
        "**Detected natural pauses**",
        "",
        "| Position | Time | Duration | Interpretation | Confidence |",
        "|---|---:|---:|---|---:|",
    ]
    labels = {"short": "short pause", "breath": "likely breath", "long": "long pause"}
    for pause in result.pauses:
        before = result.words[pause.after_word].text
        after = result.words[pause.before_word].text
        lines.append(
            f"| {before} → {after} | {pause.start_s:.2f}s | "
            f"{pause.duration_s:.2f}s | {labels.get(pause.kind, pause.kind)} | "
            f"{pause.confidence:.0%} |"
        )
    return "\n".join(lines)


def words_from_result(result: RecognitionResult | dict[str, Any]) -> Iterable[WordSpan]:
    if isinstance(result, dict):
        result = RecognitionResult.from_dict(result)
    return result.words
