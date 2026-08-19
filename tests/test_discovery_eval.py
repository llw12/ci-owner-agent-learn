from hashlib import sha256
import json
from pathlib import Path

from ci_owner_agent.discovery import (
    ChunkDiscovery,
    ErrorFragment,
    LogChunk,
)
from ci_owner_agent.discovery_eval import evaluate_dataset


class EvalFakeDiscoveryModel:
    model_name = "eval-fake-model"

    def extract_errors(self, chunk: LogChunk) -> ChunkDiscovery:
        return ChunkDiscovery(
            errors=[
                ErrorFragment(
                    title="Assertion failed",
                    category="assertion",
                    primary_message="AssertionError",
                    start_line=2,
                    end_line=3,
                )
            ]
        )


def test_evaluate_dataset_matches_candidate_to_manifest_anchor(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "eval"
    logs_dir = data_dir / "logs"
    logs_dir.mkdir(parents=True)
    log_bytes = b"setup\nAssertionError: expected 1\n at test\nsummary\n"
    (logs_dir / "sample.log").write_bytes(log_bytes)
    manifest = {
        "purpose": "test",
        "gold_semantics": [],
        "cases": [
            {
                "file": "sample.log",
                "expected_count": 1,
                "case": "assertion",
                "difficulty": "easy",
                "notes": "one assertion",
                "anchors": [
                    {
                        "start_line": 2,
                        "end_line": 3,
                        "contains": "AssertionError",
                    }
                ],
                "sha256": sha256(log_bytes).hexdigest(),
                "lines": 4,
            }
        ],
    }
    (data_dir / "golden_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )

    summary = evaluate_dataset(data_dir, EvalFakeDiscoveryModel())

    assert len(summary.cases) == 1
    assert summary.cases[0].log_file == "sample.log"
    assert summary.cases[0].expected_count == 1
    assert summary.cases[0].predicted_count == 1
    assert summary.cases[0].matched_gold == 1
    assert summary.cases[0].false_negatives == 0
    assert summary.cases[0].false_positives == 0
    assert summary.cases[0].recall == 1.0
    assert summary.cases[0].precision == 1.0
    assert summary.micro_recall == 1.0
    assert summary.micro_precision == 1.0


def test_evaluate_dataset_counts_prediction_on_success_case_as_false_positive(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "eval"
    logs_dir = data_dir / "logs"
    logs_dir.mkdir(parents=True)
    log_bytes = b"setup\nAssertionError in a passing test\n at test\nFinished: SUCCESS\n"
    (logs_dir / "success.log").write_bytes(log_bytes)
    manifest = {
        "purpose": "test",
        "gold_semantics": [],
        "cases": [
            {
                "file": "success.log",
                "expected_count": 0,
                "case": "negative_success",
                "difficulty": "hard",
                "notes": "error-like output in a successful build",
                "anchors": [{"line": 4, "text": "Finished: SUCCESS"}],
                "sha256": sha256(log_bytes).hexdigest(),
                "lines": 4,
            }
        ],
    }
    (data_dir / "golden_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )

    summary = evaluate_dataset(data_dir, EvalFakeDiscoveryModel())

    assert summary.cases[0].matched_gold == 0
    assert summary.cases[0].false_negatives == 0
    assert summary.cases[0].false_positives == 1
    assert summary.cases[0].recall == 1.0
    assert summary.cases[0].precision == 0.0
