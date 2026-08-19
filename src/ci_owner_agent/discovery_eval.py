"""Explicit local evaluation entry point for the Phase 3 discoverer."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
from typing import Sequence

from pydantic import BaseModel, ConfigDict, Field

from ci_owner_agent.discovery import (
    DiscoveryModel,
    JsonlTraceSink,
    LLMDiscoverer,
    LocalLogReader,
    NoOpTraceSink,
    OpenAICompatibleDiscoveryModel,
    TraceSink,
)
from ci_owner_agent.domain import BuildContext, ErrorCandidate


class EvalModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class GoldAnchor(EvalModel):
    line: int | None = None
    text: str | None = None
    start_line: int | None = None
    end_line: int | None = None
    contains: str | None = None


class GoldCase(EvalModel):
    file: str
    expected_count: int
    case: str
    difficulty: str
    notes: str
    anchors: list[GoldAnchor] = Field(default_factory=list)
    sha256: str
    lines: int


class GoldManifest(EvalModel):
    purpose: str
    gold_semantics: list[str] = Field(default_factory=list)
    cases: list[GoldCase] = Field(default_factory=list)


class EvalCaseResult(EvalModel):
    log_file: str
    expected_count: int
    predicted_count: int
    matched_gold: int
    false_negatives: int
    false_positives: int
    recall: float
    precision: float


class EvalSummary(EvalModel):
    cases: list[EvalCaseResult]
    micro_recall: float
    micro_precision: float


def _candidate_matches_anchor(
    candidate: ErrorCandidate, anchor: GoldAnchor
) -> bool:
    anchor_start = anchor.start_line if anchor.start_line is not None else anchor.line
    anchor_end = anchor.end_line if anchor.end_line is not None else anchor.line
    if anchor_start is None or anchor_end is None:
        return False

    overlaps = any(
        span.start_line <= anchor_end and span.end_line >= anchor_start
        for span in candidate.log_spans
    )
    if not overlaps:
        return False

    needle = anchor.contains or anchor.text
    if not needle:
        return True
    candidate_text = "\n".join(
        [
            candidate.primary_message,
            candidate.test_name or "",
            *(span.excerpt or "" for span in candidate.log_spans),
        ]
    )
    return needle.casefold() in candidate_text.casefold()


def _count_matches(
    gold_case: GoldCase, candidates: list[ErrorCandidate]
) -> int:
    if gold_case.expected_count == 0:
        return 0
    unmatched_candidates = set(range(len(candidates)))
    matched = 0
    for anchor in gold_case.anchors:
        for candidate_index in sorted(unmatched_candidates):
            if _candidate_matches_anchor(candidates[candidate_index], anchor):
                unmatched_candidates.remove(candidate_index)
                matched += 1
                break
    return min(matched, gold_case.expected_count)


def evaluate_dataset(
    data_dir: str | Path,
    model: DiscoveryModel,
    *,
    chunk_lines: int = 400,
    overlap_lines: int = 40,
    trace_sink: TraceSink | None = None,
    capture_content: bool = False,
) -> EvalSummary:
    data_path = Path(data_dir)
    manifest = GoldManifest.model_validate_json(
        (data_path / "golden_manifest.json").read_text(encoding="utf-8")
    )
    results: list[EvalCaseResult] = []

    for gold_case in manifest.cases:
        log_path = data_path / "logs" / gold_case.file
        actual_hash = sha256(log_path.read_bytes()).hexdigest()
        if actual_hash != gold_case.sha256:
            raise ValueError(f"log hash mismatch for {gold_case.file}")

        discoverer = LLMDiscoverer(
            LocalLogReader(
                chunk_lines=chunk_lines,
                overlap_lines=overlap_lines,
            ),
            model,
            trace_sink=trace_sink or NoOpTraceSink(),
            capture_content=capture_content,
        )
        candidates = discoverer.discover(
            BuildContext(
                build_id=f"eval:{gold_case.file}",
                job_name="discover-eval",
                log_ref=str(log_path),
            )
        )
        matched = _count_matches(gold_case, candidates)
        predicted = len(candidates)
        expected = gold_case.expected_count
        results.append(
            EvalCaseResult(
                log_file=gold_case.file,
                expected_count=expected,
                predicted_count=predicted,
                matched_gold=matched,
                false_negatives=expected - matched,
                false_positives=predicted - matched,
                recall=matched / expected if expected else 1.0,
                precision=(
                    matched / predicted
                    if predicted
                    else (1.0 if expected == 0 else 0.0)
                ),
            )
        )

    total_expected = sum(result.expected_count for result in results)
    total_predicted = sum(result.predicted_count for result in results)
    total_matched = sum(result.matched_gold for result in results)
    return EvalSummary(
        cases=results,
        micro_recall=(
            total_matched / total_expected if total_expected else 1.0
        ),
        micro_precision=(
            total_matched / total_predicted
            if total_predicted
            else (1.0 if total_expected == 0 else 0.0)
        ),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run local Discover LLM evaluation")
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--chunk-lines", type=int, default=400)
    parser.add_argument("--overlap-lines", type=int, default=40)
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--capture-content", action="store_true")
    args = parser.parse_args(argv)

    trace_sink: TraceSink = (
        JsonlTraceSink(args.trace) if args.trace else NoOpTraceSink()
    )
    summary = evaluate_dataset(
        args.data,
        OpenAICompatibleDiscoveryModel.from_env(),
        chunk_lines=args.chunk_lines,
        overlap_lines=args.overlap_lines,
        trace_sink=trace_sink,
        capture_content=args.capture_content,
    )
    print(json.dumps(summary.model_dump(mode="json"), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
