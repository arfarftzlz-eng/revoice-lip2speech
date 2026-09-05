"""Accuracy-first language reranking for VSR N-best hypotheses.

The language service may only score hypotheses emitted by the visual speech
recognizer and never creates or edits transcript text. Accuracy-first mode uses
Qwen's live Top-5 choice directly; visual-safe fusion remains available as an
internal compatibility mode.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Sequence


DEFAULT_RERANK_API_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_RERANK_MODEL = "qwen3.7-flash"
DEFAULT_SELECTION_MODE = "accuracy_first"
DEFAULT_LANGUAGE_WEIGHT = 1.00
DEFAULT_MAX_VISUAL_DROP_PER_TOKEN = 0.10
DEFAULT_MAX_CTC_DROP_PER_TOKEN = 0.75
DEFAULT_MIN_CONFIDENCE = 0.60
DEFAULT_OVERRIDE_MARGIN = 0.00
DEFAULT_TIMEOUT_S = 5.0


@dataclass(frozen=True)
class RecognitionCandidate:
    """One unchanged hypothesis emitted by the USR 2.0 beam search."""

    rank: int
    text: str
    token_ids: tuple[int, ...]
    score: float
    component_scores: dict[str, float] = field(default_factory=dict)

    @property
    def token_count(self) -> int:
        return max(1, len(self.token_ids))

    def public_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "text": self.text,
            "score": self.score,
            "component_scores": dict(self.component_scores),
            "token_count": self.token_count,
        }


@dataclass(frozen=True)
class RerankConfig:
    enabled: bool = False
    api_key: str = ""
    api_base: str = DEFAULT_RERANK_API_BASE
    model: str = DEFAULT_RERANK_MODEL
    selection_mode: str = DEFAULT_SELECTION_MODE
    language_weight: float = DEFAULT_LANGUAGE_WEIGHT
    max_visual_drop_per_token: float = DEFAULT_MAX_VISUAL_DROP_PER_TOKEN
    max_ctc_drop_per_token: float = DEFAULT_MAX_CTC_DROP_PER_TOKEN
    min_confidence: float = DEFAULT_MIN_CONFIDENCE
    override_margin: float = DEFAULT_OVERRIDE_MARGIN
    timeout_s: float = DEFAULT_TIMEOUT_S


@dataclass
class RerankDecision:
    top1: RecognitionCandidate
    selected: RecognitionCandidate
    eligible_ranks: list[int]
    visual_drops: dict[int, float]
    candidate_texts: dict[int, str] = field(default_factory=dict)
    language_scores: dict[int, float] = field(default_factory=dict)
    fused_scores: dict[int, float] = field(default_factory=dict)
    confidence: float = 0.0
    model: str = ""
    selection_mode: str = DEFAULT_SELECTION_MODE
    reason: str = ""
    error: str = ""

    @property
    def changed(self) -> bool:
        return self.selected.rank != self.top1.rank

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_rank": self.top1.rank,
            "original_text": self.top1.text,
            "selected_rank": self.selected.rank,
            "selected_text": self.selected.text,
            "changed": self.changed,
            "eligible_ranks": list(self.eligible_ranks),
            "visual_drops_per_token": dict(self.visual_drops),
            "candidate_texts": dict(self.candidate_texts),
            "language_scores": dict(self.language_scores),
            "fused_scores": dict(self.fused_scores),
            "confidence": self.confidence,
            "model": self.model,
            "selection_mode": self.selection_mode,
            "reason": self.reason,
            "error": self.error,
        }


def clean_api_key(value: str) -> str:
    """Remove common paste artefacts without ever persisting the secret."""
    key = str(value or "").strip().strip('"').strip("'").strip()
    if key.lower().startswith("bearer "):
        key = key[7:].strip()
    return key


def chat_completions_url(api_base: str) -> str:
    base = str(api_base or DEFAULT_RERANK_API_BASE).strip().rstrip("/")
    parsed = urllib.parse.urlparse(base)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("Language API Base URL must be an absolute HTTP(S) URL.")
    if parsed.path.rstrip("/").endswith("/chat/completions"):
        return base
    return f"{base}/chat/completions"


def _score_drop_per_token(
    top1: RecognitionCandidate,
    candidate: RecognitionCandidate,
    score_name: str | None = None,
) -> float | None:
    if score_name is None:
        top_score = top1.score
        candidate_score = candidate.score
    else:
        if score_name not in top1.component_scores or score_name not in candidate.component_scores:
            return None
        top_score = top1.component_scores[score_name]
        candidate_score = candidate.component_scores[score_name]
    denominator = max(top1.token_count, candidate.token_count)
    return max(0.0, float(top_score) - float(candidate_score)) / denominator


def visually_eligible_candidates(
    candidates: Sequence[RecognitionCandidate],
    *,
    max_visual_drop_per_token: float = DEFAULT_MAX_VISUAL_DROP_PER_TOKEN,
    max_ctc_drop_per_token: float = DEFAULT_MAX_CTC_DROP_PER_TOKEN,
) -> tuple[list[RecognitionCandidate], dict[int, float]]:
    """Apply hard total-score and CTC gates before any language request."""
    if not candidates:
        raise ValueError("At least one recognition candidate is required.")
    top1 = candidates[0]
    eligible = []
    visual_drops = {}
    for candidate in candidates:
        total_drop = _score_drop_per_token(top1, candidate) or 0.0
        ctc_drop = _score_drop_per_token(top1, candidate, "ctc")
        total_ok = candidate.rank == top1.rank or total_drop <= max_visual_drop_per_token
        ctc_ok = (
            candidate.rank == top1.rank
            or ctc_drop is None
            or ctc_drop <= max_ctc_drop_per_token
        )
        if total_ok and ctc_ok:
            eligible.append(candidate)
            visual_drops[candidate.rank] = total_drop
    return eligible, visual_drops


def _language_prompt(candidates: Sequence[RecognitionCandidate]) -> tuple[str, str]:
    system = (
        "You are a constrained language reranker for English visual speech "
        "recognition. Score only how grammatical, idiomatic, and plausible each "
        "complete spoken utterance is. Do not create, edit, merge, or rewrite any "
        "candidate. Return JSON only. Give every candidate a language_score from "
        "0 to 1, select exactly one existing id, and give confidence from 0 to 1."
    )
    payload = {
        "task": "Score and select one existing candidate without rewriting.",
        "candidates": [
            {"id": candidate.rank, "text": candidate.text}
            for candidate in candidates
        ],
        "required_json": {
            "scores": [{"id": 1, "language_score": 0.0}],
            "selected_id": 1,
            "confidence": 0.0,
        },
    }
    return system, json.dumps(payload, ensure_ascii=False)


def _decode_json_content(content: Any) -> dict[str, Any]:
    if isinstance(content, list):
        content = "".join(
            str(item.get("text", "")) if isinstance(item, dict) else str(item)
            for item in content
        )
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("Language API returned empty content.")
    clean = content.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", clean, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        clean = fenced.group(1).strip()
    try:
        value = json.loads(clean)
    except json.JSONDecodeError as error:
        raise RuntimeError("Language API returned invalid JSON content.") from error
    if not isinstance(value, dict):
        raise RuntimeError("Language API JSON must be an object.")
    return value


def _normalise_unit_score(value: Any, field_name: str) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"Language API returned an invalid {field_name}.") from error
    if 1.0 < score <= 100.0:
        score /= 100.0
    if not 0.0 <= score <= 1.0:
        raise RuntimeError(f"Language API {field_name} must be between 0 and 1.")
    return score


def _post_chat_request(
    api_base: str,
    api_key: str,
    model: str,
    candidates: Sequence[RecognitionCandidate],
    *,
    timeout_s: float,
) -> dict[str, Any]:
    system, user = _language_prompt(candidates)
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.1,
        "max_tokens": 220,
        "stream": False,
        "response_format": {"type": "json_object"},
    }
    host = urllib.parse.urlparse(api_base).netloc.lower()
    if "aliyuncs.com" in host:
        payload["enable_thinking"] = False

    url = chat_completions_url(api_base)
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            response_data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        detail = error.read(500).decode("utf-8", errors="replace")
        if api_key:
            detail = detail.replace(api_key, "[REDACTED]")
        raise RuntimeError(f"Language API returned HTTP {error.code}: {detail}") from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"Could not reach language API: {error.reason}") from error
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError("Language API returned an invalid response.") from error

    try:
        content = response_data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as error:
        raise RuntimeError("Language API response contains no assistant message.") from error
    return _decode_json_content(content)


def request_language_scores(
    candidates: Sequence[RecognitionCandidate],
    *,
    api_key: str,
    api_base: str = DEFAULT_RERANK_API_BASE,
    model: str = DEFAULT_RERANK_MODEL,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> tuple[dict[int, float], int, float]:
    """Request deterministic scores while rejecting IDs not emitted by VSR."""
    key = clean_api_key(api_key)
    if not key:
        raise ValueError("Enter a language-model API key before enabling reranking.")
    model = str(model or DEFAULT_RERANK_MODEL).strip()
    if not model:
        raise ValueError("Language model name cannot be empty.")
    if not candidates:
        raise ValueError("There are no candidates to rerank.")

    response = _post_chat_request(
        api_base or DEFAULT_RERANK_API_BASE,
        key,
        model,
        candidates,
        timeout_s=timeout_s,
    )
    allowed_ids = {candidate.rank for candidate in candidates}
    scores: dict[int, float] = {}
    raw_scores = response.get("scores")
    if not isinstance(raw_scores, list):
        raise RuntimeError("Language API JSON contains no scores array.")
    for item in raw_scores:
        if not isinstance(item, dict):
            continue
        try:
            candidate_id = int(item.get("id"))
        except (TypeError, ValueError):
            continue
        if candidate_id in allowed_ids:
            scores[candidate_id] = _normalise_unit_score(
                item.get("language_score"), "language_score"
            )
    if set(scores) != allowed_ids:
        raise RuntimeError("Language API did not score every eligible candidate.")
    try:
        selected_id = int(response.get("selected_id"))
    except (TypeError, ValueError) as error:
        raise RuntimeError("Language API returned an invalid selected_id.") from error
    if selected_id not in allowed_ids:
        raise RuntimeError("Language API selected an ID outside the eligible candidates.")
    confidence = _normalise_unit_score(response.get("confidence"), "confidence")
    return scores, selected_id, confidence


def rerank_candidates(
    candidates: Sequence[RecognitionCandidate],
    config: RerankConfig,
) -> RerankDecision:
    """Select from live N-best candidates, falling back to Top-1 on API failure."""
    if not candidates:
        raise ValueError("At least one recognition candidate is required.")
    top1 = candidates[0]
    if not config.enabled:
        return RerankDecision(
            top1=top1,
            selected=top1,
            eligible_ranks=[top1.rank],
            visual_drops={top1.rank: 0.0},
            candidate_texts={candidate.rank: candidate.text for candidate in candidates},
            reason="Language reranking is off.",
        )

    mode = str(config.selection_mode or DEFAULT_SELECTION_MODE).strip().lower()
    if mode not in {"accuracy_first", "visual_safe"}:
        mode = DEFAULT_SELECTION_MODE
    all_visual_drops = {
        candidate.rank: (_score_drop_per_token(top1, candidate) or 0.0)
        for candidate in candidates
    }
    eligible, gated_visual_drops = visually_eligible_candidates(
        candidates,
        max_visual_drop_per_token=config.max_visual_drop_per_token,
        max_ctc_drop_per_token=config.max_ctc_drop_per_token,
    )
    if mode == "visual_safe" and len(eligible) < 2:
        return RerankDecision(
            top1=top1,
            selected=top1,
            eligible_ranks=[candidate.rank for candidate in eligible],
            visual_drops=gated_visual_drops,
            candidate_texts={candidate.rank: candidate.text for candidate in candidates},
            model=config.model,
            selection_mode=mode,
            reason="No lower-ranked candidate passed the visual evidence gate.",
        )

    try:
        # Every request scores the current video's live Top-5. No dataset cache is
        # consulted by online recognition.
        language_scores, api_selected_id, confidence = request_language_scores(
            candidates,
            api_key=config.api_key,
            api_base=config.api_base,
            model=config.model,
            timeout_s=config.timeout_s,
        )
        if mode == "accuracy_first":
            selected = next(
                candidate for candidate in candidates if candidate.rank == api_selected_id
            )
            return RerankDecision(
                top1=top1,
                selected=selected,
                eligible_ranks=[candidate.rank for candidate in candidates],
                visual_drops=all_visual_drops,
                candidate_texts={candidate.rank: candidate.text for candidate in candidates},
                language_scores=language_scores,
                fused_scores=dict(language_scores),
                confidence=confidence,
                model=config.model,
                selection_mode=mode,
                reason=(
                    f"Accuracy-first mode used Qwen's candidate {selected.rank} "
                    "choice from the live Top-5."
                ),
            )

        visual_drops = gated_visual_drops
        top_language_score = language_scores[top1.rank]
        fused_scores = {
            candidate.rank: (
                -visual_drops[candidate.rank]
                + config.language_weight
                * (language_scores[candidate.rank] - top_language_score)
            )
            for candidate in eligible
        }
        winner = max(eligible, key=lambda item: (fused_scores[item.rank], -item.rank))
        if confidence < config.min_confidence:
            selected = top1
            reason = (
                f"Language confidence {confidence:.0%} was below the "
                f"{config.min_confidence:.0%} safety threshold."
            )
        elif winner.rank != top1.rank and fused_scores[winner.rank] <= config.override_margin:
            selected = top1
            reason = "The fused advantage was too small to override the visual Top-1."
        else:
            selected = winner
            if selected.rank == top1.rank:
                reason = "Visual and language evidence kept the original Top-1."
            else:
                reason = (
                    f"Candidate {selected.rank} passed the visual gate and won the "
                    "combined visual-language score."
                )
        if api_selected_id != winner.rank:
            reason += " Local visual fusion overruled the language-only preference."
        return RerankDecision(
            top1=top1,
            selected=selected,
            eligible_ranks=[candidate.rank for candidate in eligible],
            visual_drops=visual_drops,
            candidate_texts={candidate.rank: candidate.text for candidate in candidates},
            language_scores=language_scores,
            fused_scores=fused_scores,
            confidence=confidence,
            model=config.model,
            selection_mode=mode,
            reason=reason,
        )
    except Exception as error:
        fallback_candidates = candidates if mode == "accuracy_first" else eligible
        fallback_drops = all_visual_drops if mode == "accuracy_first" else gated_visual_drops
        return RerankDecision(
            top1=top1,
            selected=top1,
            eligible_ranks=[candidate.rank for candidate in fallback_candidates],
            visual_drops=fallback_drops,
            candidate_texts={candidate.rank: candidate.text for candidate in candidates},
            model=config.model,
            selection_mode=mode,
            reason="Language reranking failed safely; the visual Top-1 was retained.",
            error=f"{type(error).__name__}: {error}",
        )


def format_rerank_markdown(decision: RerankDecision | None) -> str:
    if decision is None:
        return "ℹ️ **Language reranking:** Off. The visual Top-1 was used."
    if decision.error:
        return (
            "⚠️ **Language reranking:** Visual Top-1 retained. "
            f"{decision.error}"
        )
    if not decision.language_scores:
        return f"ℹ️ **Language reranking:** {decision.reason}"

    outcome = (
        f"selected candidate {decision.selected.rank}"
        if decision.changed
        else "kept candidate 1"
    )
    if decision.selection_mode == "accuracy_first":
        lines = [
            f"✅ **Language reranking · accuracy first:** {outcome} · "
            f"{decision.model} · confidence {decision.confidence:.0%}.",
            "",
            "| Candidate | Visual drop/token | Qwen score | Text |",
            "|---:|---:|---:|---|",
        ]
        for rank in decision.eligible_ranks:
            text = decision.candidate_texts.get(rank, "Beam candidate")
            lines.append(
                f"| {rank} | {decision.visual_drops[rank]:.3f} | "
                f"{decision.language_scores[rank]:.2f} | {text} |"
            )
        lines.extend(["", decision.reason])
        return "\n".join(lines)

    lines = [
        f"✅ **Language reranking · visual safety:** {outcome} · {decision.model} · "
        f"confidence {decision.confidence:.0%}.",
        "",
        "| Candidate | Visual drop/token | Language | Fused | Text |",
        "|---:|---:|---:|---:|---|",
    ]
    for rank in decision.eligible_ranks:
        text = decision.candidate_texts.get(rank, "Eligible beam candidate")
        lines.append(
            f"| {rank} | {decision.visual_drops[rank]:.3f} | "
            f"{decision.language_scores[rank]:.2f} | "
            f"{decision.fused_scores[rank]:+.3f} | {text} |"
        )
    lines.extend(["", decision.reason])
    return "\n".join(lines)


def test_language_connection(api_key: str, api_base: str, model: str) -> str:
    """Make one tiny, billable scoring request without retaining the key."""
    candidates = [
        RecognitionCandidate(1, "THIS PROJECT CAN QUIT MY LIPS", (1, 2, 3), -1.0),
        RecognitionCandidate(2, "THIS PROJECT CAN READ MY LIPS", (1, 2, 4), -1.1),
    ]
    try:
        scores, selected_id, confidence = request_language_scores(
            candidates,
            api_key=api_key,
            api_base=api_base or DEFAULT_RERANK_API_BASE,
            model=model or DEFAULT_RERANK_MODEL,
        )
    except Exception as error:
        return f"❌ Language API test failed: {type(error).__name__}: {error}"
    return (
        f"✅ Language API connected · {model or DEFAULT_RERANK_MODEL} · "
        f"selected sample candidate {selected_id} · confidence {confidence:.0%} · "
        f"scores {json.dumps(scores, ensure_ascii=False)}."
    )
