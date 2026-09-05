"""Prosody-aware, GPU-independent speech synthesis for the Gradio demo."""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np


def _configure_ffmpeg() -> Optional[str]:
    executable = shutil.which("ffmpeg")
    if executable:
        return executable
    try:
        import imageio_ffmpeg
    except ImportError:
        return None
    executable = imageio_ffmpeg.get_ffmpeg_exe()
    os.environ["PATH"] = str(Path(executable).parent) + os.pathsep + os.environ.get("PATH", "")
    return executable


_FFMPEG = _configure_ffmpeg()

import edge_tts
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message="Couldn't find ffmpeg or avconv")
    from pydub import AudioSegment

from prosody import PauseEvent, RecognitionResult


if _FFMPEG:
    AudioSegment.converter = _FFMPEG
    AudioSegment.ffmpeg = _FFMPEG


DEFAULT_VOICE = "en-US-AriaNeural"
DEFAULT_ELEVENLABS_VOICE_ID = "JBFqnCBsd6RMkjVDRZzb"
DEFAULT_MINIMAX_VOICE_ID = "English_Trustworthy_Man"
DEFAULT_MINIMAX_GLOBAL_VOICE_ID = "English_Trustworth_Man"
_RELAYROUTER_HOSTS = {
    "api.relayrouter.ai",
}
MINIMAX_REGION_AUTO = "auto"
MINIMAX_REGION_GLOBAL = "global"
MINIMAX_REGION_CHINA = "china"
MINIMAX_REGIONS = [
    MINIMAX_REGION_AUTO,
    MINIMAX_REGION_GLOBAL,
    MINIMAX_REGION_CHINA,
]
MINIMAX_REGION_CHOICES = [
    ("Auto detect", MINIMAX_REGION_AUTO),
    ("Global (minimax.io)", MINIMAX_REGION_GLOBAL),
    ("Mainland China (minimax.cn)", MINIMAX_REGION_CHINA),
]
MODE_AUTO = "auto"
MODE_MINIMAX = "minimax"
MODE_ELEVENLABS = "elevenlabs"
# Backward-compatible name retained for callers of the earlier implementation.
MODE_EXPRESSIVE = MODE_ELEVENLABS
MODE_STANDARD = "edge"
TTS_MODES = [MODE_AUTO, MODE_MINIMAX, MODE_ELEVENLABS, MODE_STANDARD]
TTS_MODE_CHOICES = [
    ("Natural (automatic)", MODE_AUTO),
    ("Expressive AI (MiniMax Speech-2.8 HD)", MODE_MINIMAX),
    ("Expressive AI (ElevenLabs v3)", MODE_ELEVENLABS),
    ("Standard (Edge TTS)", MODE_STANDARD),
]
STYLE_CONSERVATIVE = "conservative"
STYLE_NATURAL = "natural"
STYLE_EXPRESSIVE = "expressive"
STYLE_LEVELS = [STYLE_CONSERVATIVE, STYLE_NATURAL, STYLE_EXPRESSIVE]
STYLE_LEVEL_CHOICES = [
    ("Conservative", STYLE_CONSERVATIVE),
    ("Natural", STYLE_NATURAL),
    ("Expressive", STYLE_EXPRESSIVE),
]
_DEFAULT_OUTPUT_DIRECTORY = tempfile.TemporaryDirectory(prefix="usr2_tts_")


@dataclass
class SynthesisResult:
    path: str
    provider: str
    note: str = ""
    expressive_script: str = ""


_LEGACY_MODE_ALIASES = {
    "natural (automatic)": MODE_AUTO,
    "expressive ai (minimax speech-2.8 hd)": MODE_MINIMAX,
    "expressive ai (elevenlabs v3)": MODE_ELEVENLABS,
    "standard (edge tts)": MODE_STANDARD,
    "自然（自动）": MODE_AUTO,
    "富有表现力的人工智能（minimax 语音-2.8高清）": MODE_MINIMAX,
    "富有表现力的人工智能（elevenlabs v3）": MODE_ELEVENLABS,
    "标准（edge tts）": MODE_STANDARD,
}
_LEGACY_STYLE_ALIASES = {
    "conservative": STYLE_CONSERVATIVE,
    "natural": STYLE_NATURAL,
    "expressive": STYLE_EXPRESSIVE,
    "保守的": STYLE_CONSERVATIVE,
    "自然的": STYLE_NATURAL,
    "富有表现力的": STYLE_EXPRESSIVE,
}
_LEGACY_REGION_ALIASES = {
    "auto detect": MINIMAX_REGION_AUTO,
    "global (minimax.io)": MINIMAX_REGION_GLOBAL,
    "mainland china (minimax.cn)": MINIMAX_REGION_CHINA,
    "自动检测": MINIMAX_REGION_AUTO,
    "全球（minimax.io）": MINIMAX_REGION_GLOBAL,
    "中国大陆（minimax.cn）": MINIMAX_REGION_CHINA,
}


def _normalize_choice(value: str, aliases: dict[str, str]) -> str:
    clean_value = str(value or "").strip()
    return aliases.get(clean_value.lower(), clean_value)


def _clean_minimax_api_key(value: Optional[str]) -> str:
    """Accept a pasted secret with accidental quotes or a Bearer prefix."""
    clean_value = (value or "").strip()
    if clean_value.lower().startswith("bearer "):
        clean_value = clean_value[7:].strip()
    if (
        len(clean_value) >= 2
        and clean_value[0] == clean_value[-1]
        and clean_value[0] in {"'", '"'}
    ):
        clean_value = clean_value[1:-1].strip()
    return clean_value


def _clean_minimax_api_base(value: Optional[str]) -> str:
    """Validate and normalize an official or MiniMax-compatible API base URL."""
    clean_value = (value or "").strip().strip("'\"").rstrip("/")
    if not clean_value:
        return ""

    parsed = urllib.parse.urlsplit(clean_value)
    hostname = (parsed.hostname or "").lower()
    local_hosts = {"127.0.0.1", "localhost", "::1"}
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
        raise RuntimeError(
            "API Base URL must be a complete URL such as https://provider.example/v1"
        )
    if parsed.scheme.lower() != "https" and hostname not in local_hosts:
        raise RuntimeError("API Base URL must use HTTPS (HTTP is allowed only locally)")
    if parsed.username or parsed.password:
        raise RuntimeError("Do not put credentials in the API Base URL")
    if parsed.query or parsed.fragment:
        raise RuntimeError("API Base URL cannot contain a query string or fragment")

    # People often paste a complete MiniMax endpoint instead of its base. Reduce
    # either supported endpoint back to the shared base so both validation and
    # synthesis can use it.
    path = parsed.path.rstrip("/")
    for suffix in ("/get_voice", "/t2a_v2"):
        if path.lower().endswith(suffix):
            path = path[: -len(suffix)].rstrip("/")
            break
    return urllib.parse.urlunsplit(
        (parsed.scheme.lower(), parsed.netloc, path, "", "")
    ).rstrip("/")


def _minimax_request_url(api_base: str, path: str) -> str:
    """Join a base URL with a MiniMax path without duplicating ``/v1``."""
    base = api_base.rstrip("/")
    request_path = "/" + path.lstrip("/")
    if base.lower().endswith("/v1") and request_path.lower().startswith("/v1/"):
        request_path = request_path[3:]
    return f"{base}{request_path}"


def _relayrouter_origin(api_base: str) -> str:
    """Return the RelayRouter origin, or an empty string for other providers."""
    parsed = urllib.parse.urlsplit(api_base)
    if (parsed.hostname or "").lower() not in _RELAYROUTER_HOSTS:
        return ""
    return urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, "", "", "")).rstrip(
        "/"
    )


def normalize_text_for_tts(raw_text: str) -> str:
    """Apply whitespace, casing, and terminal-punctuation cleanup only."""
    text = re.sub(r"\s+", " ", (raw_text or "").strip())
    if not text:
        return ""
    if text.isupper():
        text = text.lower()
        text = text[0].upper() + text[1:]
        text = re.sub(r"\bi\b", "I", text)
    elif text[0].isalpha():
        text = text[0].upper() + text[1:]
    if text[-1] not in ".?!":
        text += "."
    return text


def _coerce_recognition(
    recognition: RecognitionResult | dict[str, Any] | None,
) -> RecognitionResult | None:
    if recognition is None:
        return None
    if isinstance(recognition, RecognitionResult):
        return recognition
    if isinstance(recognition, dict):
        return RecognitionResult.from_dict(recognition)
    raise TypeError("recognition must be a RecognitionResult, dictionary, or None")


def _spoken_words(result: RecognitionResult) -> list[str]:
    words = [word.text for word in result.words]
    if not words:
        return []
    if all(word.upper() == word for word in words):
        words = [word.lower() for word in words]
        words = ["I" if word == "i" else word for word in words]
        words[0] = words[0][0].upper() + words[0][1:]
    return words


def build_expressive_script(
    text: str,
    recognition: RecognitionResult | dict[str, Any] | None,
    style: str = "Natural",
) -> str:
    """Create a v3-style script with pauses but no generated vocal gestures."""
    result = _coerce_recognition(recognition)
    if result is None or not result.words or result.alignment_status != "ok":
        return normalize_text_for_tts(text)

    words = _spoken_words(result)
    pauses = {pause.after_word: pause for pause in result.pauses}
    output: list[str] = []
    for word_index, word in enumerate(words):
        output.append(word)
        pause = pauses.get(word_index)
        if pause is None:
            continue
        output.append("[long pause]" if pause.kind == "long" else "[short pause]")

    script = " ".join(output).strip()
    if script and script[-1] not in ".?!":
        script += "."
    return script


def build_minimax_script(
    text: str,
    recognition: RecognitionResult | dict[str, Any] | None,
    style: str = "Natural",
) -> str:
    """Create a MiniMax script containing only speech and measured pauses.

    Provider-side interjection tags are deliberately excluded: depending on
    voice and context, a requested breath may become a voiced hesitation. Any
    breath texture is mixed locally after synthesis and therefore cannot turn
    into an invented word such as "uh" or "um".
    """
    result = _coerce_recognition(recognition)
    if result is None or not result.words or result.alignment_status != "ok":
        return normalize_text_for_tts(text)

    words = _spoken_words(result)
    pauses = {pause.after_word: pause for pause in result.pauses}
    output: list[str] = []
    for word_index, word in enumerate(words):
        output.append(word)
        pause = pauses.get(word_index)
        if pause is None:
            continue
        duration_s = float(np.clip(pause.duration_s, 0.01, 99.99))
        output.append(f"<#{duration_s:.2f}#>")

    script = " ".join(output).strip()
    if script and script[-1] not in ".?!":
        script += "."
    return script


def _result_directory(output_dir: Optional[Union[str, Path]]) -> Path:
    if output_dir is None:
        return Path(_DEFAULT_OUTPUT_DIRECTORY.name)
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _safe_unlink(path: Path) -> None:
    """Best-effort cleanup for short-lived Windows media file locks."""
    for attempt in range(3):
        try:
            path.unlink(missing_ok=True)
            return
        except PermissionError:
            if attempt < 2:
                time.sleep(0.05 * (attempt + 1))


def _load_mp3(path: Path) -> AudioSegment:
    # Supplying a decoder codec keeps Pydub from requiring a separate ffprobe
    # binary. imageio-ffmpeg bundles ffmpeg itself but not ffprobe on Windows.
    return AudioSegment.from_file(path, format="mp3", codec="mp3")


def _export_audio(
    audio: AudioSegment,
    path: Path,
    format_name: str,
    **kwargs: Any,
) -> None:
    """Export audio and promptly release Pydub's returned file handle."""
    handle = audio.export(path, format=format_name, **kwargs)
    if hasattr(handle, "close"):
        handle.close()


def _edge_speech(text: str, output_path: Path, voice: str) -> None:
    edge_tts.Communicate(normalize_text_for_tts(text), voice).save_sync(str(output_path))
    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise RuntimeError("The Edge TTS service returned an empty audio file")


def _minimax_endpoints(
    region: str,
    api_base: Optional[str] = None,
) -> list[tuple[str, str, str]]:
    """Return official endpoint, default English voice, and display label."""
    region = _normalize_choice(region, _LEGACY_REGION_ALIASES)
    custom_base = _clean_minimax_api_base(api_base) or _clean_minimax_api_base(
        os.environ.get("MINIMAX_API_BASE", "")
    )
    custom_voice = os.environ.get("MINIMAX_VOICE_ID", "").strip()
    if custom_base:
        relayrouter_origin = _relayrouter_origin(custom_base)
        if relayrouter_origin:
            return [
                (
                    f"{relayrouter_origin}/minimax",
                    custom_voice or DEFAULT_MINIMAX_VOICE_ID,
                    "RelayRouter · MiniMax native route",
                )
            ]
        return [
            (
                custom_base.rstrip("/"),
                custom_voice or DEFAULT_MINIMAX_VOICE_ID,
                "Custom endpoint",
            )
        ]

    if region not in MINIMAX_REGIONS:
        raise ValueError(f"Unsupported MiniMax account region: {region}")
    global_endpoint = (
        "https://api.minimax.io",
        custom_voice or DEFAULT_MINIMAX_GLOBAL_VOICE_ID,
        "Global",
    )
    china_endpoint = (
        "https://api.minimax.cn",
        custom_voice or DEFAULT_MINIMAX_VOICE_ID,
        "Mainland China",
    )
    if region == MINIMAX_REGION_GLOBAL:
        return [global_endpoint]
    if region == MINIMAX_REGION_CHINA:
        return [china_endpoint]
    return [global_endpoint, china_endpoint]


def _minimax_json_request(
    api_base: str,
    path: str,
    payload: dict[str, Any],
    api_key: str,
    *,
    timeout: int,
) -> dict[str, Any]:
    request = urllib.request.Request(
        _minimax_request_url(api_base, path),
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            response_data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        detail = error.read(500).decode("utf-8", errors="replace")
        raise RuntimeError(f"MiniMax returned HTTP {error.code}: {detail}") from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"Could not reach MiniMax: {error.reason}") from error
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError("MiniMax returned an invalid JSON response") from error

    if not isinstance(response_data, dict):
        raise RuntimeError("MiniMax returned an unexpected response")
    base_response = response_data.get("base_resp") or {}
    status_code = base_response.get("status_code")
    if status_code != 0:
        status_message = base_response.get("status_msg") or "unknown error"
        raise RuntimeError(f"MiniMax request failed ({status_code}): {status_message}")
    return response_data


def _authorized_json_get(url: str, api_key: str, *, timeout: int) -> dict[str, Any]:
    """Make a bearer-authenticated JSON GET request without provider assumptions."""
    request = urllib.request.Request(
        url,
        method="GET",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            response_data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        detail = error.read(500).decode("utf-8", errors="replace")
        raise RuntimeError(f"Gateway returned HTTP {error.code}: {detail}") from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"Could not reach gateway: {error.reason}") from error
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError("Gateway returned an invalid JSON response") from error
    if not isinstance(response_data, dict):
        raise RuntimeError("Gateway returned an unexpected response")
    return response_data


def _can_retry_minimax_region(error: Exception) -> bool:
    text = str(error).lower()
    return any(
        marker in text
        for marker in (
            "http 401",
            "http 403",
            "invalid token",
            "api key",
            "apikey",
            "unauthor",
            "authorization",
            "permission",
            "could not reach",
            "1004",
        )
    )


def validate_minimax_key(
    api_key: Optional[str],
    region: str = MINIMAX_REGION_AUTO,
    api_base: Optional[str] = None,
) -> str:
    """Validate a key with MiniMax's non-billable voice-list endpoint."""
    clean_key = _clean_minimax_api_key(api_key)
    if not clean_key:
        return "❌ Enter a MiniMax API key first."

    failures: list[str] = []
    try:
        custom_base = _clean_minimax_api_base(api_base) or _clean_minimax_api_base(
            os.environ.get("MINIMAX_API_BASE", "")
        )
        custom_endpoint = bool(custom_base)
        relayrouter_origin = _relayrouter_origin(custom_base) if custom_base else ""
        if relayrouter_origin:
            response = _authorized_json_get(
                f"{relayrouter_origin}/v1/models",
                clean_key,
                timeout=30,
            )
            model_ids = {
                str(model.get("id", "")).strip().lower()
                for model in response.get("data", [])
                if isinstance(model, dict)
            }
            speech_models = sorted(
                model for model in model_ids if model.startswith("speech-")
            )
            if "speech-2.8-hd" in model_ids:
                return "✅ RelayRouter key verified · MiniMax speech-2.8-hd is available."
            if speech_models:
                return (
                    "⚠️ RelayRouter key verified, but speech-2.8-hd is unavailable. "
                    f"Speech models on this key: {', '.join(speech_models[:6])}."
                )
            return (
                "⚠️ RelayRouter key verified, but this key's model group exposes no "
                "MiniMax speech model. Enable speech-2.8-hd for the key first."
            )
        endpoints = _minimax_endpoints(region, api_base)
    except Exception as error:
        return f"❌ TTS connection configuration error · {error}"
    for index, (endpoint_base, _voice_id, label) in enumerate(endpoints):
        try:
            response = _minimax_json_request(
                endpoint_base,
                "/v1/get_voice",
                {"voice_type": "system"},
                clean_key,
                timeout=30,
            )
            voice_count = len(response.get("system_voice") or [])
            return f"✅ Key verified · {label} · {voice_count} system voices available."
        except Exception as error:
            failures.append(f"{label}: {error}")
            if index + 1 >= len(endpoints) or not _can_retry_minimax_region(error):
                break
    detail = "; ".join(failures)
    if failures and all("(1004)" in failure for failure in failures):
        if custom_endpoint:
            return (
                "❌ Key rejected by the custom endpoint. Confirm that the Base URL "
                "and key come from the same provider, and that the service supports "
                "MiniMax-compatible /v1/get_voice and /v1/t2a_v2 routes."
            )
        return (
            "❌ Key rejected by MiniMax. Create an API Secret in the MiniMax API "
            "Platform (not the consumer app), then paste only the key—without "
            "quotes or a Bearer prefix."
        )
    if len(detail) > 420:
        detail = detail[:417] + "..."
    return f"❌ MiniMax verification failed · {detail}"


def _minimax_speech(
    script: str,
    output_path: Path,
    api_key: Optional[str] = None,
    region: str = MINIMAX_REGION_AUTO,
    api_base: Optional[str] = None,
) -> dict[str, Any]:
    """Call MiniMax's synchronous text-to-audio API and save its hex MP3."""
    api_key = _clean_minimax_api_key(api_key) or _clean_minimax_api_key(
        os.environ.get("MINIMAX_API_KEY", "")
    )
    if not api_key:
        raise RuntimeError(
            "MINIMAX_API_KEY is not configured. Set it before launching the app, "
            "or use Natural (automatic) mode."
        )

    emotion = os.environ.get("MINIMAX_EMOTION", "").strip().lower()
    supported_emotions = {
        "happy",
        "sad",
        "angry",
        "fearful",
        "disgusted",
        "surprised",
        "calm",
        "fluent",
        "whipser",
    }
    if emotion:
        if emotion not in supported_emotions:
            raise RuntimeError(
                "MINIMAX_EMOTION must be one of: "
                + ", ".join(sorted(supported_emotions))
            )

    failures: list[str] = []
    endpoints = _minimax_endpoints(region, api_base)
    for index, (api_base, voice_id, label) in enumerate(endpoints):
        if not voice_id:
            raise RuntimeError("MINIMAX_VOICE_ID cannot be empty")
        voice_setting: dict[str, Any] = {
            "voice_id": voice_id,
            "speed": 1.0,
            "vol": 1.0,
            "pitch": 0,
        }
        if emotion:
            voice_setting["emotion"] = emotion
        payload = {
            "model": "speech-2.8-hd",
            "text": script,
            "stream": False,
            "language_boost": "English",
            "output_format": "hex",
            "voice_setting": voice_setting,
            "audio_setting": {
                "sample_rate": 32000,
                "bitrate": 128000,
                "format": "mp3",
                "channel": 1,
            },
            "subtitle_enable": False,
            "aigc_watermark": False,
        }
        try:
            response_data = _minimax_json_request(
                api_base,
                "/v1/t2a_v2",
                payload,
                api_key,
                timeout=120,
            )
            break
        except Exception as error:
            failures.append(f"{label}: {error}")
            if index + 1 >= len(endpoints) or not _can_retry_minimax_region(error):
                raise RuntimeError("; ".join(failures)) from error
    else:
        raise RuntimeError("; ".join(failures))

    data = response_data.get("data") or {}
    if data.get("status") != 2:
        raise RuntimeError(
            f"MiniMax synthesis did not finish (status={data.get('status')})"
        )
    encoded_audio = data.get("audio")
    if not isinstance(encoded_audio, str) or not encoded_audio:
        raise RuntimeError("MiniMax returned no audio data")
    try:
        audio_bytes = bytes.fromhex(encoded_audio)
    except ValueError as error:
        raise RuntimeError("MiniMax returned malformed hex audio data") from error
    if not audio_bytes:
        raise RuntimeError("MiniMax returned an empty audio file")
    output_path.write_bytes(audio_bytes)
    extra_info = dict(response_data.get("extra_info") or {})
    extra_info["_minimax_region"] = label
    return extra_info


def _elevenlabs_speech(script: str, output_path: Path) -> None:
    api_key = os.environ.get("ELEVENLABS_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "ELEVENLABS_API_KEY is not configured. Set it before launching the app, "
            "or use Natural (automatic) mode."
        )
    voice_id = os.environ.get(
        "ELEVENLABS_VOICE_ID", DEFAULT_ELEVENLABS_VOICE_ID
    ).strip()
    encoded_voice_id = urllib.parse.quote(voice_id, safe="")
    url = (
        f"https://api.elevenlabs.io/v1/text-to-speech/{encoded_voice_id}"
        "?output_format=mp3_44100_128"
    )
    payload = json.dumps({"text": script, "model_id": "eleven_v3"}).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=payload,
        method="POST",
        headers={
            "xi-api-key": api_key,
            "Content-Type": "application/json",
            "Accept": "audio/mpeg",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            output_path.write_bytes(response.read())
    except urllib.error.HTTPError as error:
        detail = error.read(500).decode("utf-8", errors="replace")
        raise RuntimeError(f"ElevenLabs returned HTTP {error.code}: {detail}") from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"Could not reach ElevenLabs: {error.reason}") from error
    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise RuntimeError("ElevenLabs returned an empty audio file")


def _audio_from_array(samples: np.ndarray, frame_rate: int = 44100) -> AudioSegment:
    samples = np.clip(samples, -32767.0, 32767.0).astype(np.int16)
    return AudioSegment(
        data=samples.tobytes(), sample_width=2, frame_rate=frame_rate, channels=1
    )


def _room_tone(duration_ms: int, level_dbfs: float = -52.0) -> AudioSegment:
    """Create quiet, deterministic broadband room tone instead of digital zero."""
    duration_ms = max(0, int(duration_ms))
    if duration_ms == 0:
        return AudioSegment.silent(duration=0, frame_rate=44100)
    frame_rate = 44100
    sample_count = max(1, int(frame_rate * duration_ms / 1000))
    rng = np.random.default_rng(1701 + duration_ms)
    noise = rng.standard_normal(sample_count)
    noise = np.convolve(noise, np.ones(9) / 9.0, mode="same")
    rms = float(np.sqrt(np.mean(noise * noise))) or 1.0
    target_rms = 32767.0 * (10.0 ** (level_dbfs / 20.0))
    return _audio_from_array(noise * (target_rms / rms), frame_rate)


def _breath_sound(duration_ms: int, level_dbfs: float = -38.0) -> AudioSegment:
    """Generate an unvoiced, restrained breath texture from filtered noise."""
    duration_ms = int(np.clip(duration_ms, 140, 480))
    frame_rate = 44100
    sample_count = max(1, int(frame_rate * duration_ms / 1000))
    rng = np.random.default_rng(9109 + duration_ms)
    noise = rng.standard_normal(sample_count)
    low = np.convolve(noise, np.ones(121) / 121.0, mode="same")
    airy = np.convolve(noise - low, np.ones(5) / 5.0, mode="same")
    x = np.linspace(0.0, 1.0, sample_count)
    envelope = (np.sin(np.pi * x) ** 1.7) * (0.75 + 0.25 * x)
    airy *= envelope
    rms = float(np.sqrt(np.mean(airy * airy))) or 1.0
    target_rms = 32767.0 * (10.0 ** (level_dbfs / 20.0))
    return _audio_from_array(airy * (target_rms / rms), frame_rate).fade_in(60).fade_out(90)


def _atempo_chain(tempo: float) -> str:
    factors: list[float] = []
    remaining = max(0.05, float(tempo))
    while remaining > 2.0:
        factors.append(2.0)
        remaining /= 2.0
    while remaining < 0.5:
        factors.append(0.5)
        remaining /= 0.5
    if not math.isclose(remaining, 1.0, rel_tol=0.01):
        factors.append(remaining)
    if not factors:
        factors.append(1.0)
    return ",".join(f"atempo={factor:.6f}" for factor in factors)


def _fit_duration(audio: AudioSegment, target_ms: int) -> AudioSegment:
    """Pitch-preservingly fit audio to a target duration with bundled FFmpeg."""
    target_ms = max(80, int(target_ms))
    if len(audio) == 0 or abs(len(audio) - target_ms) <= 20:
        return audio[:target_ms]
    if not _FFMPEG:
        raise RuntimeError("FFmpeg is required to preserve the detected speaking rhythm")
    tempo = len(audio) / target_ms
    with tempfile.TemporaryDirectory(prefix="usr2_atempo_") as temporary:
        input_path = Path(temporary) / "input.wav"
        output_path = Path(temporary) / "output.wav"
        _export_audio(audio, input_path, "wav")
        process = subprocess.run(
            [
                _FFMPEG, "-y", "-loglevel", "error", "-i", str(input_path),
                "-filter:a", _atempo_chain(tempo), str(output_path),
            ],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        if process.returncode != 0 or not output_path.is_file():
            raise RuntimeError(f"FFmpeg timing adjustment failed: {process.stderr.strip()}")
        fitted = AudioSegment.from_file(output_path, format="wav")
    if len(fitted) > target_ms:
        return fitted[:target_ms]
    if len(fitted) < target_ms:
        return fitted + _room_tone(target_ms - len(fitted))
    return fitted


def _with_room_tone(audio: AudioSegment) -> AudioSegment:
    base = _room_tone(len(audio), -52.0).set_channels(audio.channels).set_frame_rate(
        audio.frame_rate
    )
    return base.overlay(audio)


def _pause_render_mode(pause: PauseEvent) -> str:
    """Choose only a non-vocal breath or the provider's existing word tail."""
    if pause.duration_s <= 0.30:
        return "short"
    if pause.blank_confidence >= 0.82 and pause.inactivity >= 0.80:
        return "breath"
    return "tail"


def _overlay_natural_breaths(
    audio: AudioSegment,
    recognition: RecognitionResult | None,
    style: str,
) -> AudioSegment:
    """Add pure breath only where evidence favors it; otherwise keep the word tail."""
    if recognition is None:
        return audio

    output = audio
    for pause in recognition.pauses:
        if _pause_render_mode(pause) != "breath":
            continue

        pause_duration_ms = round(pause.duration_s * 1000)
        breath_duration_ms = round(
            float(np.clip(pause_duration_ms * 0.55, 180, 480))
        )
        position_ms = round(
            pause.start_s * 1000
            + max(0, pause_duration_ms - breath_duration_ms) * 0.35
        )
        level_by_style = {
            STYLE_CONSERVATIVE: -42.0,
            STYLE_NATURAL: -39.0,
            STYLE_EXPRESSIVE: -35.5,
        }
        level_dbfs = level_by_style.get(style.lower(), -39.0)
        breath = _breath_sound(breath_duration_ms, level_dbfs)
        breath = breath.set_channels(audio.channels).set_frame_rate(audio.frame_rate)
        output = output.overlay(breath, position=max(0, position_ms))
    return output


def _finish_cloud_audio(
    audio: AudioSegment,
    recognition: RecognitionResult | None,
    style: str,
    preserve_timing: bool,
) -> AudioSegment:
    """Fit cloud speech to the visual timeline, then add only local ambience."""
    if preserve_timing and recognition is not None and recognition.video_duration_s > 0:
        audio = _fit_duration(audio, round(recognition.video_duration_s * 1000))
    audio = _with_room_tone(audio)
    if preserve_timing:
        audio = _overlay_natural_breaths(audio, recognition, style)
    return audio


def _phrase_ranges(result: RecognitionResult) -> list[tuple[str, int, int]]:
    words = _spoken_words(result)
    if not words or not result.words:
        return []
    pause_by_word = {pause.after_word: pause for pause in result.pauses}
    ranges: list[tuple[str, int, int]] = []
    first_word = 0
    for word_index in range(len(words)):
        pause = pause_by_word.get(word_index)
        if pause is None:
            continue
        start_ms = round(result.words[first_word].start_frame * result.frame_duration_s * 1000)
        end_ms = round(pause.start_s * 1000)
        ranges.append((" ".join(words[first_word : word_index + 1]), start_ms, end_ms))
        first_word = word_index + 1
    if first_word < len(words):
        start_ms = round(result.words[first_word].start_frame * result.frame_duration_s * 1000)
        last_word = result.words[-1]
        end_ms = round((last_word.end_frame + 1) * result.frame_duration_s * 1000)
        ranges.append((" ".join(words[first_word:]), start_ms, end_ms))
    return [item for item in ranges if item[2] > item[1] and item[0].strip()]


def _edge_with_reconstructed_timing(
    text: str,
    recognition: RecognitionResult | None,
    directory: Path,
    voice: str,
    style: str,
    preserve_timing: bool,
) -> AudioSegment:
    if (
        not preserve_timing
        or recognition is None
        or recognition.alignment_status != "ok"
        or recognition.video_duration_s <= 0
    ):
        source_path = directory / f"edge_source_{uuid.uuid4().hex}.mp3"
        try:
            _edge_speech(text, source_path, voice)
            return _with_room_tone(_load_mp3(source_path))
        finally:
            _safe_unlink(source_path)

    total_ms = max(100, round(recognition.video_duration_s * 1000))
    canvas = _room_tone(total_ms)
    phrase_ranges = _phrase_ranges(recognition)
    if not phrase_ranges:
        source_path = directory / f"edge_source_{uuid.uuid4().hex}.mp3"
        try:
            _edge_speech(text, source_path, voice)
            speech = _fit_duration(_load_mp3(source_path), total_ms)
            return canvas.overlay(speech)
        finally:
            _safe_unlink(source_path)

    for phrase_text, start_ms, end_ms in phrase_ranges:
        source_path = directory / f"edge_phrase_{uuid.uuid4().hex}.mp3"
        try:
            _edge_speech(phrase_text, source_path, voice)
            phrase = _fit_duration(_load_mp3(source_path), end_ms - start_ms)
            fade_ms = min(35, max(0, len(phrase) // 6))
            if fade_ms:
                phrase = phrase.fade_in(fade_ms).fade_out(fade_ms)
            canvas = canvas.overlay(phrase, position=max(0, start_ms))
        finally:
            _safe_unlink(source_path)

    return _overlay_natural_breaths(canvas[:total_ms], recognition, style)


def synthesize_speech(
    text: str,
    output_dir: Optional[Union[str, Path]] = None,
    *,
    voice: str = DEFAULT_VOICE,
    mode: str = MODE_AUTO,
    style: str = "Natural",
    recognition: RecognitionResult | dict[str, Any] | None = None,
    preserve_timing: bool = True,
    minimax_api_key: Optional[str] = None,
    minimax_region: str = MINIMAX_REGION_AUTO,
    minimax_api_base: Optional[str] = None,
) -> SynthesisResult:
    """Synthesize speech with domestic-first expressive-provider fallback."""
    mode = _normalize_choice(mode, _LEGACY_MODE_ALIASES)
    style = _normalize_choice(style, _LEGACY_STYLE_ALIASES)
    minimax_region = _normalize_choice(minimax_region, _LEGACY_REGION_ALIASES)
    normalized = normalize_text_for_tts(text)
    if not normalized:
        raise ValueError("There is no recognized text to read aloud.")
    if mode not in TTS_MODES:
        raise ValueError(f"Unsupported TTS mode: {mode}")
    if style not in STYLE_LEVELS:
        raise ValueError(f"Unsupported expression level: {style}")

    directory = _result_directory(output_dir)
    result = _coerce_recognition(recognition)
    output_path = directory / f"speech_{uuid.uuid4().hex}.mp3"
    source_path = directory / f"expressive_source_{uuid.uuid4().hex}.mp3"
    minimax_available = bool(
        (minimax_api_key or "").strip()
        or os.environ.get("MINIMAX_API_KEY", "").strip()
    )
    elevenlabs_available = bool(os.environ.get("ELEVENLABS_API_KEY", "").strip())
    provider_failures: list[str] = []

    try:
        if mode == MODE_MINIMAX or (mode == MODE_AUTO and minimax_available):
            script = build_minimax_script(normalized, result, style)
            try:
                extra_info = _minimax_speech(
                    script,
                    source_path,
                    api_key=minimax_api_key,
                    region=minimax_region,
                    api_base=minimax_api_base,
                )
                audio = _load_mp3(source_path)
                audio = _finish_cloud_audio(audio, result, style, preserve_timing)
                _export_audio(audio, output_path, "mp3", bitrate="128k")
                region_label = (
                    extra_info.get("_minimax_region", "verified endpoint")
                    if isinstance(extra_info, dict)
                    else "verified endpoint"
                )
                return SynthesisResult(
                    str(output_path.resolve()),
                    "MiniMax Speech-2.8 HD",
                    f"{region_label}. MiniMax preserved measured pauses; sparse "
                    "unvoiced breaths and room tone were mixed locally.",
                    script,
                )
            except Exception as error:
                if mode == MODE_MINIMAX:
                    raise
                provider_failures.append(f"MiniMax unavailable: {error}")
                _safe_unlink(source_path)

        if mode == MODE_ELEVENLABS or (mode == MODE_AUTO and elevenlabs_available):
            script = build_expressive_script(normalized, result, style)
            try:
                _elevenlabs_speech(script, source_path)
                audio = _load_mp3(source_path)
                audio = _finish_cloud_audio(audio, result, style, preserve_timing)
                _export_audio(audio, output_path, "mp3", bitrate="128k")
                return SynthesisResult(
                    str(output_path.resolve()),
                    "ElevenLabs v3",
                    "Expressive AI TTS used visual/CTC timing without generated "
                    "vocal interjections.",
                    script,
                )
            except Exception as error:
                if mode == MODE_ELEVENLABS:
                    raise
                provider_failures.append(
                    f"ElevenLabs unavailable ({type(error).__name__})"
                )
                _safe_unlink(source_path)

        audio = _edge_with_reconstructed_timing(
            normalized, result, directory, voice, style, preserve_timing
        )
        _export_audio(audio, output_path, "mp3", bitrate="128k")
        note = ""
        if provider_failures:
            note = "; ".join(provider_failures) + (
                "; used Edge TTS with reconstructed timing, subtle breaths, and room tone."
            )
        elif mode == MODE_AUTO and not minimax_available and not elevenlabs_available:
            note = (
                "No cloud TTS key is configured; used Edge TTS with reconstructed "
                "timing, subtle breaths, and room tone. Set MINIMAX_API_KEY to enable "
                "the domestic expressive provider."
            )
        return SynthesisResult(str(output_path.resolve()), "Edge TTS", note)
    except Exception:
        _safe_unlink(output_path)
        raise
    finally:
        _safe_unlink(source_path)


def text_to_speech(
    text: str,
    output_dir: Optional[Union[str, Path]] = None,
    voice: str = DEFAULT_VOICE,
) -> str:
    """Backward-compatible plain-text TTS wrapper."""
    return synthesize_speech(
        text,
        output_dir,
        voice=voice,
        mode=MODE_STANDARD,
        preserve_timing=False,
    ).path
