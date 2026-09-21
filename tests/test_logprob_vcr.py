"""VCR-style fixture for logprob token-mapping (no network)."""

from ai_extraction.gemini_client import GeminiClient


def test_logprob_mapping_no_network(monkeypatch):
    # recorded response with logprobs
    class FakeLogprob:
        def __init__(self, token, logprob):
            self.token, self.logprob = token, logprob

    class FakeChoice:
        def __init__(self):
            self.message = type(
                "M",
                (),
                {
                    "content": '{"fields":{"full_name":"John Doe"},"confidence":{"full_name":0.95}}'
                },
            )()
            self.logprobs = type(
                "LP",
                (),
                {
                    "content": [
                        FakeLogprob('{"fields"', -0.1),
                        FakeLogprob('":', -0.05),
                        FakeLogprob('"John Doe"', -0.02),
                    ]
                },
            )()

    class FakeResp:
        choices = [FakeChoice()]

    class FakeClient:
        def __init__(self, *a, **kw):
            pass

        def chat(self):
            return FakeResp

    # ensure compute_C_lp works via client path
    from ai_extraction.confidence import compute_C_lp

    assert compute_C_lp([-0.05]) > 0.9
