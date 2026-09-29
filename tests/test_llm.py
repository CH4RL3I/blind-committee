import pytest
from conftest import ScriptedBackend, view_json

from agentfund.llm import LLMValidationError, extract_json, structured
from agentfund.schemas import AnalystView


def test_valid_output_first_try():
    b = ScriptedBackend([view_json()])
    v, attempts = structured(b, AnalystView, "sys", "prompt")
    assert v.stance == "bullish" and attempts == 1


def test_invalid_json_then_schema_error_then_valid_retries_with_feedback():
    bad_schema = '{"stance": "moonshot", "confidence": 3, "reasoning": "x"}'
    b = ScriptedBackend(["not json at all", bad_schema, view_json("bearish", 0.4)])
    v, attempts = structured(b, AnalystView, "sys", "prompt", retries=2)
    assert (v.stance, attempts) == ("bearish", 3)
    assert "previous reply was invalid" in b.calls[1][1]  # error fed back to the model


def test_gives_up_after_retry_budget():
    b = ScriptedBackend(["nope"] * 3)
    with pytest.raises(LLMValidationError):
        structured(b, AnalystView, "sys", "prompt", retries=2)
    assert len(b.calls) == 3


def test_extract_json_tolerates_fences_and_chatter():
    assert extract_json('Sure!\n```json\n{"a": 1}\n```') == {"a": 1}
