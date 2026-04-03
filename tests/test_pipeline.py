import json

from src.eval import ConversationResult, TurnResult
from src.pipeline import load_results, save_results


def make_dummy_result() -> ConversationResult:
    """Creates a dummy ConversationResult for testing"""
    cr = ConversationResult(record_id="test_record")
    cr.turns = [
        TurnResult(
            turn_index=0,
            question="what was revenue in 2009?",
            predicted="206588.0",
            predicted_raw="STEPS:\n1. found it\nANSWER: 206588.0",
            gold_executed=206588.0,
            gold_program="206588",
            em=True,
            num_match=True,
            parse_error=False,
        )
    ]
    return cr

def test_save_and_load_roundtrip(tmp_path):
    """Save results to disk and reload — data should be identical."""
    path    = tmp_path / "results.json"
    results = [make_dummy_result()]

    save_results(results, path)
    loaded = load_results(path)

    assert len(loaded) == 1
    assert loaded[0].record_id == "test_record"
    assert len(loaded[0].turns) == 1
    assert loaded[0].turns[0].predicted == "206588.0"
    assert loaded[0].turns[0].num_match is True


def test_save_creates_valid_json(tmp_path):
    """Output file should be valid parseable JSON."""
    path = tmp_path / "results.json"
    save_results([make_dummy_result()], path)

    with open(path) as f:
        raw = json.load(f)

    assert isinstance(raw, list)
    assert raw[0]["record_id"] == "test_record"
    assert len(raw[0]["turns"]) == 1


def test_load_empty_file(tmp_path):
    """Empty results list should roundtrip cleanly."""
    path = tmp_path / "empty.json"
    save_results([], path)
    loaded = load_results(path)
    assert loaded == []
