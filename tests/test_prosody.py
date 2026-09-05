import json
import math
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from prosody import (
    PauseEvent,
    RecognitionResult,
    TokenSpan,
    WordSpan,
    ctc_forced_align,
    detect_pauses,
    merge_token_spans_into_words,
)
from tts import (
    MINIMAX_REGION_AUTO,
    MINIMAX_REGION_CHOICES,
    MODE_AUTO,
    STYLE_LEVEL_CHOICES,
    TTS_MODE_CHOICES,
    _atempo_chain,
    _breath_sound,
    _minimax_speech,
    _overlay_natural_breaths,
    _pause_render_mode,
    _room_tone,
    build_expressive_script,
    build_minimax_script,
    synthesize_speech,
    validate_minimax_key,
)


class UIChoiceTests(unittest.TestCase):
    def test_translatable_labels_use_stable_machine_values(self):
        for choices in (
            TTS_MODE_CHOICES,
            STYLE_LEVEL_CHOICES,
            MINIMAX_REGION_CHOICES,
        ):
            for label, value in choices:
                self.assertNotEqual(label, value)
                self.assertRegex(value, r"^[a-z]+$")


class CTCAlignmentTests(unittest.TestCase):
    def test_forced_alignment_finds_token_spikes(self):
        probabilities = np.full((8, 4), 0.01, dtype=np.float64)
        path = [0, 1, 0, 0, 0, 2, 0, 0]
        for frame, label in enumerate(path):
            probabilities[frame, label] = 0.97
        spans = ctc_forced_align(
            np.log(probabilities), [1, 2], ["<blank>", "▁HELLO", "▁WORLD", "<eos>"]
        )
        self.assertEqual([(span.start_frame, span.end_frame) for span in spans], [(1, 1), (5, 5)])

    def test_repeated_tokens_require_an_intervening_blank(self):
        probabilities = np.full((6, 3), 0.01, dtype=np.float64)
        path = [0, 1, 0, 1, 0, 0]
        for frame, label in enumerate(path):
            probabilities[frame, label] = 0.98
        spans = ctc_forced_align(
            np.log(probabilities), [1, 1], ["<blank>", "A", "<eos>"]
        )
        self.assertEqual([span.start_frame for span in spans], [1, 3])

    def test_sentencepiece_units_merge_into_words(self):
        spans = [
            TokenSpan(1, "▁HEL", 1, 1, 0.9),
            TokenSpan(2, "LO", 2, 3, 0.8),
            TokenSpan(3, "▁WORLD", 7, 8, 0.95),
        ]
        words = merge_token_spans_into_words(spans)
        self.assertEqual([word.text for word in words], ["HELLO", "WORLD"])
        self.assertEqual((words[0].start_frame, words[0].end_frame), (1, 3))


class PauseDetectionTests(unittest.TestCase):
    def test_low_motion_inter_word_gap_becomes_pause(self):
        words = [
            WordSpan("HELLO", 0, 1, 0.9, 0, 0),
            WordSpan("WORLD", 7, 8, 0.9, 1, 1),
        ]
        probabilities = np.full((10, 3), 0.01, dtype=np.float64)
        probabilities[:, 0] = 0.97
        activity = np.zeros(10, dtype=np.float64)
        pauses = detect_pauses(words, np.log(probabilities), activity, 0.04)
        self.assertEqual(len(pauses), 1)
        self.assertAlmostEqual(pauses[0].duration_s, 0.20)

    def test_high_motion_gap_is_rejected(self):
        words = [
            WordSpan("HELLO", 0, 1, 0.9, 0, 0),
            WordSpan("WORLD", 7, 8, 0.9, 1, 1),
        ]
        probabilities = np.full((10, 3), 0.01, dtype=np.float64)
        probabilities[:, 0] = 0.97
        activity = np.ones(10, dtype=np.float64)
        self.assertEqual(detect_pauses(words, np.log(probabilities), activity, 0.04), [])

    def test_three_frame_short_pause_is_kept(self):
        words = [
            WordSpan("I", 0, 1, 0.9, 0, 0),
            WordSpan("KNOW", 5, 7, 0.9, 1, 1),
        ]
        probabilities = np.full((9, 3), 0.01, dtype=np.float64)
        probabilities[:, 0] = 0.05
        probabilities[2:5, 0] = 0.93
        activity = np.ones(9, dtype=np.float64) * 0.85
        activity[2:5] = 0.08

        pauses = detect_pauses(words, np.log(probabilities), activity, 0.04)

        self.assertEqual(len(pauses), 1)
        self.assertAlmostEqual(pauses[0].duration_s, 0.12)
        self.assertEqual(pauses[0].kind, "short")

    def test_moving_gap_edges_do_not_hide_still_core(self):
        words = [
            WordSpan("PLEASE", 0, 1, 0.9, 0, 0),
            WordSpan("WAIT", 10, 12, 0.9, 1, 1),
        ]
        probabilities = np.full((14, 3), 0.01, dtype=np.float64)
        probabilities[:, 0] = 0.08
        probabilities[3:9, 0] = 0.94
        activity = np.ones(14, dtype=np.float64) * 0.90
        activity[3:9] = 0.06

        pauses = detect_pauses(words, np.log(probabilities), activity, 0.04)

        self.assertEqual(len(pauses), 1)
        self.assertEqual((pauses[0].start_frame, pauses[0].end_frame), (3, 8))
        self.assertAlmostEqual(pauses[0].duration_s, 0.24)

    def test_strong_boundary_evidence_recovers_ctc_compressed_pause(self):
        words = [
            WordSpan("WELL", 0, 4, 0.9, 0, 0),
            WordSpan("MAYBE", 5, 9, 0.9, 1, 1),
        ]
        probabilities = np.full((10, 3), 0.01, dtype=np.float64)
        probabilities[:, 0] = 0.05
        probabilities[3:6, 0] = 0.96
        activity = np.ones(10, dtype=np.float64) * 0.88
        activity[3:6] = 0.04

        pauses = detect_pauses(words, np.log(probabilities), activity, 0.04)

        self.assertEqual(len(pauses), 1)
        self.assertEqual((pauses[0].start_frame, pauses[0].end_frame), (3, 5))
        self.assertAlmostEqual(pauses[0].duration_s, 0.12)


class SpeechPlanningTests(unittest.TestCase):
    def _result(self):
        return RecognitionResult(
            text="I THINK WE SHOULD GO",
            words=[
                WordSpan("I", 0, 1, 0.9, 0, 0),
                WordSpan("THINK", 2, 5, 0.9, 1, 1),
                WordSpan("WE", 17, 18, 0.9, 2, 2),
                WordSpan("SHOULD", 19, 22, 0.9, 3, 3),
                WordSpan("GO", 23, 25, 0.9, 4, 4),
            ],
            pauses=[
                PauseEvent(1, 2, 6, 16, 0.24, 0.68, 0.44, "breath", 0.9, 0.95, 0.85)
            ],
            frame_duration_s=0.04,
            video_duration_s=1.2,
            alignment_status="ok",
        )

    def test_expressive_script_preserves_words_without_breath_prompt(self):
        script = build_expressive_script(self._result().text, self._result(), "Natural")
        self.assertIn("[short pause]", script)
        self.assertNotIn("[inhales", script)
        spoken = " ".join(part for part in script.split() if not part.startswith("[") and not part.endswith("]"))
        self.assertNotIn(" um ", f" {spoken.lower()} ")
        self.assertNotIn(" uh ", f" {spoken.lower()} ")

    def test_minimax_script_uses_exact_pause_without_interjection(self):
        script = build_minimax_script(self._result().text, self._result(), "Natural")
        self.assertIn("<#0.44#>", script)
        self.assertNotIn("(breath)", script)
        self.assertNotIn("(inhale)", script)
        self.assertNotIn("(emm)", script)
        self.assertNotIn(" um ", f" {script.lower()} ")
        self.assertNotIn(" uh ", f" {script.lower()} ")

    def test_conservative_minimax_script_keeps_exact_silence(self):
        script = build_minimax_script(
            self._result().text, self._result(), "Conservative"
        )
        self.assertIn("<#0.44#>", script)
        self.assertNotIn("(breath)", script)

    def test_state_round_trip(self):
        original = self._result()
        restored = RecognitionResult.from_dict(original.to_dict())
        self.assertEqual(restored.text, original.text)
        self.assertEqual(restored.pauses[0].kind, "breath")

    def test_generated_ambience_has_expected_duration_and_is_not_silent(self):
        room = _room_tone(400)
        breath = _breath_sound(300)
        self.assertLessEqual(abs(len(room) - 400), 1)
        self.assertLessEqual(abs(len(breath) - 300), 1)
        self.assertTrue(math.isfinite(room.dBFS))
        self.assertTrue(math.isfinite(breath.dBFS))

    def test_local_breath_overlay_is_unvoiced_and_keeps_timeline(self):
        quiet = _room_tone(1200, -60.0)
        mixed = _overlay_natural_breaths(quiet, self._result(), "Natural")
        self.assertEqual(len(mixed), len(quiet))
        self.assertGreater(mixed[240:680].rms, quiet[240:680].rms)

        conservative = _overlay_natural_breaths(quiet, self._result(), "Conservative")
        self.assertGreater(conservative[240:680].rms, quiet[240:680].rms)
        self.assertLess(conservative[240:680].rms, mixed[240:680].rms)

    def test_pause_with_residual_speech_keeps_provider_tail(self):
        pause = PauseEvent(
            0, 1, 4, 13, 0.16, 0.56, 0.40, "breath", 0.74, 0.68, 0.72
        )
        self.assertEqual(_pause_render_mode(pause), "tail")
        result = RecognitionResult(
            text="STAY HERE",
            words=[
                WordSpan("STAY", 0, 3, 0.9, 0, 0),
                WordSpan("HERE", 14, 18, 0.9, 1, 1),
            ],
            pauses=[pause],
            video_duration_s=0.8,
            alignment_status="ok",
        )
        quiet = _room_tone(800, -60.0)
        mixed = _overlay_natural_breaths(quiet, result, "Natural")
        self.assertEqual(mixed.raw_data, quiet.raw_data)

    def test_atempo_chain_handles_large_ratios(self):
        chain = _atempo_chain(4.5)
        self.assertIn("atempo=2.000000", chain)
        factors = [float(item.split("=")[1]) for item in chain.split(",")]
        self.assertAlmostEqual(math.prod(factors), 4.5, places=5)


class MiniMaxTransportTests(unittest.TestCase):
    class _FakeResponse:
        def __init__(self, value):
            self.value = value

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

        def read(self):
            return json.dumps(self.value).encode("utf-8")

    def test_minimax_request_and_hex_response(self):
        response = {
            "data": {"audio": b"ID3-test-audio".hex(), "status": 2},
            "extra_info": {"audio_length": 500},
            "base_resp": {"status_code": 0, "status_msg": "success"},
        }
        captured = {}

        def fake_urlopen(request, timeout):
            captured["request"] = request
            captured["timeout"] = timeout
            return self._FakeResponse(response)

        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(
            os.environ,
            {
                "MINIMAX_API_KEY": "environment-key",
                "MINIMAX_API_BASE": "",
                "MINIMAX_VOICE_ID": "",
            },
            clear=False,
        ), mock.patch("urllib.request.urlopen", side_effect=fake_urlopen):
            output_path = Path(directory) / "speech.mp3"
            extra_info = _minimax_speech(
                "Hello <#0.25#> there.", output_path, api_key="Bearer test-token"
            )
            self.assertEqual(output_path.read_bytes(), b"ID3-test-audio")

        request = captured["request"]
        payload = json.loads(request.data.decode("utf-8"))
        self.assertEqual(request.full_url, "https://api.minimax.io/v1/t2a_v2")
        self.assertEqual(request.get_header("Authorization"), "Bearer test-token")
        self.assertEqual(payload["model"], "speech-2.8-hd")
        self.assertEqual(
            payload["voice_setting"]["voice_id"], "English_Trustworth_Man"
        )
        self.assertEqual(payload["language_boost"], "English")
        self.assertEqual(payload["output_format"], "hex")
        self.assertEqual(extra_info["audio_length"], 500)

    def test_minimax_service_error_is_reported(self):
        response = {
            "base_resp": {"status_code": 1004, "status_msg": "invalid token"}
        }
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(
            os.environ, {"MINIMAX_API_KEY": "test-key"}, clear=False
        ), mock.patch(
            "urllib.request.urlopen", return_value=self._FakeResponse(response)
        ):
            with self.assertRaisesRegex(RuntimeError, "invalid token"):
                _minimax_speech("Hello.", Path(directory) / "speech.mp3")

    def test_custom_base_url_is_used_without_duplicate_v1(self):
        response = {
            "data": {"audio": b"ID3-custom".hex(), "status": 2},
            "base_resp": {"status_code": 0, "status_msg": "success"},
        }
        captured = {}

        def fake_urlopen(request, timeout):
            captured["url"] = request.full_url
            return self._FakeResponse(response)

        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(
            os.environ,
            {"MINIMAX_API_BASE": "", "MINIMAX_VOICE_ID": ""},
            clear=False,
        ), mock.patch("urllib.request.urlopen", side_effect=fake_urlopen):
            _minimax_speech(
                "Hello.",
                Path(directory) / "speech.mp3",
                api_key="session-key",
                api_base="https://relay.example/v1/",
            )

        self.assertEqual(captured["url"], "https://relay.example/v1/t2a_v2")

    def test_custom_full_endpoint_is_reduced_to_base_for_validation(self):
        captured = {}

        def fake_urlopen(request, timeout):
            captured["url"] = request.full_url
            return self._FakeResponse(
                {
                    "system_voice": [],
                    "base_resp": {"status_code": 0, "status_msg": "success"},
                }
            )

        with mock.patch.dict(
            os.environ, {"MINIMAX_API_BASE": ""}, clear=False
        ), mock.patch("urllib.request.urlopen", side_effect=fake_urlopen):
            status = validate_minimax_key(
                "session-key",
                MINIMAX_REGION_AUTO,
                "https://relay.example/v1/t2a_v2",
            )

        self.assertIn("Key verified", status)
        self.assertEqual(captured["url"], "https://relay.example/v1/get_voice")

    def test_custom_remote_base_url_requires_https(self):
        status = validate_minimax_key(
            "session-key",
            MINIMAX_REGION_AUTO,
            "http://relay.example/v1",
        )
        self.assertIn("TTS connection configuration error", status)
        self.assertIn("HTTPS", status)

    def test_relayrouter_validation_uses_models_endpoint(self):
        captured = {}

        def fake_urlopen(request, timeout):
            captured["url"] = request.full_url
            captured["method"] = request.method
            captured["authorization"] = request.get_header("Authorization")
            return self._FakeResponse(
                {
                    "object": "list",
                    "data": [
                        {"id": "MiniMax-M2.7"},
                        {"id": "speech-2.8-hd"},
                    ],
                }
            )

        with mock.patch("urllib.request.urlopen", side_effect=fake_urlopen):
            status = validate_minimax_key(
                "session-key",
                MINIMAX_REGION_AUTO,
                "https://api.relayrouter.ai/v1",
            )

        self.assertIn("RelayRouter key verified", status)
        self.assertIn("speech-2.8-hd is available", status)
        self.assertEqual(captured["url"], "https://api.relayrouter.ai/v1/models")
        self.assertEqual(captured["method"], "GET")
        self.assertEqual(captured["authorization"], "Bearer session-key")

    def test_relayrouter_synthesis_uses_minimax_native_route(self):
        response = {
            "data": {"audio": b"ID3-relay".hex(), "status": 2},
            "base_resp": {"status_code": 0, "status_msg": "success"},
        }
        captured = {}

        def fake_urlopen(request, timeout):
            captured["url"] = request.full_url
            return self._FakeResponse(response)

        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(
            os.environ,
            {"MINIMAX_API_BASE": "", "MINIMAX_VOICE_ID": ""},
            clear=False,
        ), mock.patch("urllib.request.urlopen", side_effect=fake_urlopen):
            _minimax_speech(
                "Hello.",
                Path(directory) / "speech.mp3",
                api_key="session-key",
                api_base="https://api.relayrouter.ai/v1",
            )

        self.assertEqual(
            captured["url"],
            "https://api.relayrouter.ai/minimax/v1/t2a_v2",
        )

    def test_session_key_enables_automatic_minimax_without_global_state(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(
            os.environ,
            {"MINIMAX_API_KEY": "", "ELEVENLABS_API_KEY": ""},
            clear=False,
        ), mock.patch("tts._minimax_speech") as minimax_call, mock.patch(
            "tts._load_mp3", return_value=_room_tone(120)
        ):
            result = synthesize_speech(
                "Hello there.",
                directory,
                mode=MODE_AUTO,
                preserve_timing=False,
                minimax_api_key="test-key",
            )
            self.assertEqual(os.environ.get("MINIMAX_API_KEY", ""), "")

        self.assertEqual(result.provider, "MiniMax Speech-2.8 HD")
        self.assertEqual(
            minimax_call.call_args.kwargs["api_key"], "test-key"
        )

    def test_key_validation_auto_detects_mainland_endpoint(self):
        calls = []

        def fake_urlopen(request, timeout):
            calls.append(request.full_url)
            if "minimax.io" in request.full_url:
                return self._FakeResponse(
                    {
                        "base_resp": {
                            "status_code": 1004,
                            "status_msg": "invalid token",
                        }
                    }
                )
            return self._FakeResponse(
                {
                    "system_voice": [{"voice_id": "English_Trustworthy_Man"}],
                    "base_resp": {"status_code": 0, "status_msg": "success"},
                }
            )

        with mock.patch.dict(
            os.environ,
            {"MINIMAX_API_BASE": "", "MINIMAX_VOICE_ID": ""},
            clear=False,
        ), mock.patch("urllib.request.urlopen", side_effect=fake_urlopen):
            status = validate_minimax_key("session-key", MINIMAX_REGION_AUTO)

        self.assertIn("Key verified", status)
        self.assertIn("Mainland China", status)
        self.assertEqual(
            calls,
            [
                "https://api.minimax.io/v1/get_voice",
                "https://api.minimax.cn/v1/get_voice",
            ],
        )

    def test_key_validation_explains_status_1004(self):
        response = {
            "base_resp": {
                "status_code": 1004,
                "status_msg": "login fail: missing Authorization",
            }
        }
        with mock.patch.dict(
            os.environ,
            {"MINIMAX_API_BASE": "", "MINIMAX_VOICE_ID": ""},
            clear=False,
        ), mock.patch(
            "urllib.request.urlopen", return_value=self._FakeResponse(response)
        ):
            status = validate_minimax_key("not-a-real-key", MINIMAX_REGION_AUTO)

        self.assertIn("Key rejected by MiniMax", status)
        self.assertIn("API Secret", status)


if __name__ == "__main__":
    unittest.main()
