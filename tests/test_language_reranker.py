import io
import json
import unittest
import urllib.error
from unittest import mock

from language_reranker import (
    RecognitionCandidate,
    RerankConfig,
    chat_completions_url,
    request_language_scores,
    rerank_candidates,
    visually_eligible_candidates,
)


def candidate(rank, text, score, *, ctc=None, token_count=4):
    components = {} if ctc is None else {"ctc": ctc}
    return RecognitionCandidate(
        rank=rank,
        text=text,
        token_ids=tuple(range(1, token_count + 1)),
        score=score,
        component_scores=components,
    )


class FakeResponse:
    def __init__(self, value):
        self.value = value

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def read(self):
        return json.dumps(self.value).encode("utf-8")


class LanguageRerankerTests(unittest.TestCase):
    def test_chat_url_accepts_base_or_complete_endpoint(self):
        self.assertEqual(
            chat_completions_url("https://example.test/v1/"),
            "https://example.test/v1/chat/completions",
        )
        self.assertEqual(
            chat_completions_url("https://example.test/v1/chat/completions"),
            "https://example.test/v1/chat/completions",
        )

    def test_visual_gate_excludes_large_score_drop(self):
        candidates = [
            candidate(1, "TOP ONE", -1.0),
            candidate(2, "CLOSE CANDIDATE", -1.4),
            candidate(3, "DISTANT CANDIDATE", -4.0),
        ]
        eligible, drops = visually_eligible_candidates(candidates)
        self.assertEqual([item.rank for item in eligible], [1, 2])
        self.assertAlmostEqual(drops[2], 0.10)

    @mock.patch("language_reranker.request_language_scores")
    def test_language_can_select_candidate_that_passes_visual_gate(self, request_scores):
        request_scores.return_value = ({1: 0.10, 2: 0.95}, 2, 0.90)
        candidates = [
            candidate(1, "THIS PROJECT CAN QUIT MY LIPS", -1.0),
            candidate(2, "THIS PROJECT CAN READ MY LIPS", -1.4),
        ]
        decision = rerank_candidates(
            candidates,
            RerankConfig(enabled=True, api_key="secret"),
        )
        self.assertTrue(decision.changed)
        self.assertEqual(decision.selected.rank, 2)

    @mock.patch("language_reranker.request_language_scores")
    def test_low_language_confidence_keeps_visual_top1(self, request_scores):
        request_scores.return_value = ({1: 0.05, 2: 0.99}, 2, 0.40)
        candidates = [
            candidate(1, "FIRST", -1.0),
            candidate(2, "SECOND", -1.1),
        ]
        decision = rerank_candidates(
            candidates,
            RerankConfig(enabled=True, api_key="secret", selection_mode="visual_safe"),
        )
        self.assertFalse(decision.changed)
        self.assertIn("below", decision.reason)

    @mock.patch("language_reranker.request_language_scores")
    def test_api_failure_falls_back_without_raising(self, request_scores):
        request_scores.side_effect = RuntimeError("network unavailable")
        candidates = [
            candidate(1, "FIRST", -1.0),
            candidate(2, "SECOND", -1.1),
        ]
        decision = rerank_candidates(
            candidates,
            RerankConfig(enabled=True, api_key="secret"),
        )
        self.assertEqual(decision.selected.rank, 1)
        self.assertIn("network unavailable", decision.error)

    @mock.patch("language_reranker.request_language_scores")
    def test_api_scores_full_nbest_but_visual_gate_remains_authoritative(self, request_scores):
        request_scores.return_value = ({1: 0.10, 2: 0.80, 3: 1.00}, 3, 0.95)
        candidates = [
            candidate(1, "FIRST", -1.0),
            candidate(2, "SECOND", -1.2),
            candidate(3, "FLUENT BUT VISUALLY DISTANT", -9.0),
        ]
        decision = rerank_candidates(
            candidates,
            RerankConfig(enabled=True, api_key="secret", selection_mode="visual_safe"),
        )
        sent_candidates = request_scores.call_args.args[0]
        self.assertEqual([item.rank for item in sent_candidates], [1, 2, 3])
        self.assertNotEqual(decision.selected.rank, 3)

    @mock.patch("language_reranker.request_language_scores")
    def test_accuracy_first_uses_qwen_choice_even_outside_visual_gate(self, request_scores):
        request_scores.return_value = ({1: 0.10, 2: 0.20, 3: 1.00}, 3, 0.40)
        candidates = [
            candidate(1, "FIRST", -1.0),
            candidate(2, "SECOND", -1.2),
            candidate(3, "QWEN CHOICE", -9.0),
        ]
        decision = rerank_candidates(
            candidates,
            RerankConfig(enabled=True, api_key="secret"),
        )
        self.assertEqual(decision.selection_mode, "accuracy_first")
        self.assertEqual(decision.selected.rank, 3)
        self.assertTrue(decision.changed)

    @mock.patch("language_reranker._post_chat_request")
    def test_response_cannot_select_outside_candidate_set(self, post_chat):
        post_chat.return_value = {
            "scores": [
                {"id": 1, "language_score": 0.2},
                {"id": 2, "language_score": 0.8},
            ],
            "selected_id": 99,
            "confidence": 0.9,
        }
        candidates = [
            candidate(1, "FIRST", -1.0),
            candidate(2, "SECOND", -1.1),
        ]
        with self.assertRaisesRegex(RuntimeError, "outside"):
            request_language_scores(candidates, api_key="secret")

    @mock.patch("language_reranker.urllib.request.urlopen")
    def test_transport_uses_bearer_auth_and_json_mode(self, urlopen):
        content = json.dumps(
            {
                "scores": [
                    {"id": 1, "language_score": 20},
                    {"id": 2, "language_score": 90},
                ],
                "selected_id": 2,
                "confidence": 85,
            }
        )
        urlopen.return_value = FakeResponse(
            {"choices": [{"message": {"content": content}}]}
        )
        candidates = [
            candidate(1, "FIRST", -1.0),
            candidate(2, "SECOND", -1.1),
        ]
        scores, selected, confidence = request_language_scores(
            candidates,
            api_key="Bearer test-token",
            api_base="https://example.test/v1",
            model="small-model",
        )
        request = urlopen.call_args.args[0]
        body = json.loads(request.data.decode("utf-8"))
        self.assertEqual(request.full_url, "https://example.test/v1/chat/completions")
        self.assertEqual(request.headers["Authorization"], "Bearer test-token")
        self.assertNotIn("test-token", request.data.decode("utf-8"))
        self.assertEqual(body["response_format"], {"type": "json_object"})
        self.assertEqual(scores, {1: 0.2, 2: 0.9})
        self.assertEqual(selected, 2)
        self.assertEqual(confidence, 0.85)

    @mock.patch("language_reranker.urllib.request.urlopen")
    def test_http_error_redacts_key_if_gateway_echoes_it(self, urlopen):
        secret = "session-secret"
        urlopen.side_effect = urllib.error.HTTPError(
            "https://example.test/v1/chat/completions",
            401,
            "Unauthorized",
            {},
            io.BytesIO(f"bad Authorization Bearer {secret}".encode("utf-8")),
        )
        candidates = [
            candidate(1, "FIRST", -1.0),
            candidate(2, "SECOND", -1.1),
        ]
        with self.assertRaises(RuntimeError) as context:
            request_language_scores(
                candidates,
                api_key=secret,
                api_base="https://example.test/v1",
            )
        self.assertNotIn(secret, str(context.exception))
        self.assertIn("[REDACTED]", str(context.exception))


if __name__ == "__main__":
    unittest.main()
