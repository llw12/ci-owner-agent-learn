import json

import pytest
from pydantic import ValidationError

from ci_owner_agent.domain import (
    AnalysisState,
    AnalysisStage,
    BuildContext,
    Confidence,
    ErrorCandidate,
    Evidence,
    EvidenceStrength,
    EvidenceType,
    FinalError,
    FinalReport,
    InvestigationResult,
    LogSpan,
    OwnerCandidate,
    ValidationResult,
    ValidationVerdict,
)


def test_log_span_requires_one_based_ordered_lines() -> None:
    valid_span = LogSpan(start_line=10, end_line=10)

    assert valid_span.start_line == 10
    assert valid_span.end_line == 10

    with pytest.raises(ValidationError):
        LogSpan(start_line=10, end_line=9)

    with pytest.raises(ValidationError):
        LogSpan(start_line=0, end_line=1)


def test_error_candidate_forbids_investigation_fields() -> None:
    with pytest.raises(ValidationError, match="owner"):
        ErrorCandidate(
            id="error-1",
            title="Duplicate key",
            primary_message="E11000 duplicate key error",
            owner="alice",
        )


@pytest.mark.parametrize(
    ("original", "final"),
    [
        (Confidence.HIGH, Confidence.MEDIUM),
        (Confidence.MEDIUM, Confidence.MEDIUM),
        (Confidence.LOW, Confidence.NONE),
    ],
)
def test_validation_result_allows_confidence_to_stay_or_decrease(
    original: Confidence, final: Confidence
) -> None:
    investigation = InvestigationResult(
        error_id="error-1",
        hypothesis="The identifier generator changed",
        confidence=original,
    )
    validation = ValidationResult(
        error_id="error-1",
        verdict=ValidationVerdict.DOWNGRADED,
        final_confidence=final,
        root_cause_valid=False,
        ownership_valid=False,
        confidence_valid=True,
    )
    state = AnalysisState(
        build=BuildContext(
            build_id="build-1", job_name="unit-tests", log_ref="log://build-1"
        ),
        investigations={"error-1": investigation},
        validations={"error-1": validation},
    )

    assert state.validations["error-1"].final_confidence is final


def test_analysis_state_rejects_validation_confidence_upgrade() -> None:
    investigation = InvestigationResult(
        error_id="error-1",
        hypothesis="The identifier generator changed",
        confidence=Confidence.LOW,
    )
    validation = ValidationResult(
        error_id="error-1",
        verdict=ValidationVerdict.ACCEPTED,
        final_confidence=Confidence.HIGH,
        root_cause_valid=True,
        ownership_valid=True,
        confidence_valid=True,
    )

    with pytest.raises(ValidationError, match="cannot exceed"):
        AnalysisState(
            build=BuildContext(
                build_id="build-1", job_name="unit-tests", log_ref="log://build-1"
            ),
            investigations={"error-1": investigation},
            validations={"error-1": validation},
        )


def test_analysis_state_rejects_validation_without_investigation() -> None:
    validation = ValidationResult(
        error_id="error-1",
        verdict=ValidationVerdict.REJECTED,
        final_confidence=Confidence.NONE,
        root_cause_valid=False,
        ownership_valid=False,
        confidence_valid=False,
    )

    with pytest.raises(ValidationError, match="requires a matching investigation"):
        AnalysisState(
            build=BuildContext(
                build_id="build-1", job_name="unit-tests", log_ref="log://build-1"
            ),
            validations={"error-1": validation},
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("original_confidence", Confidence.LOW),
        (
            "owner_candidate",
            OwnerCandidate(identity="alice", reason="Changed the affected code"),
        ),
    ],
)
def test_validation_result_forbids_investigator_owned_fields(
    field: str, value: object
) -> None:
    with pytest.raises(ValidationError, match=field):
        ValidationResult(
            error_id="error-1",
            verdict=ValidationVerdict.ACCEPTED,
            final_confidence=Confidence.LOW,
            root_cause_valid=True,
            ownership_valid=True,
            confidence_valid=True,
            **{field: value},
        )


def test_mutable_defaults_are_isolated_between_instances() -> None:
    first_candidate = ErrorCandidate(
        id="error-1", title="First", primary_message="first failure"
    )
    second_candidate = ErrorCandidate(
        id="error-2", title="Second", primary_message="second failure"
    )
    first_candidate.log_spans.append(LogSpan(start_line=1, end_line=1))

    build = BuildContext(
        build_id="build-1",
        job_name="unit-tests",
        log_ref="jenkins://unit-tests/build-1/console",
    )
    first_state = AnalysisState(build=build)
    second_state = AnalysisState(build=build)
    first_state.investigations["error-1"] = InvestigationResult(
        error_id="error-1", hypothesis="The identifier generator changed"
    )

    assert second_candidate.log_spans == []
    assert second_state.investigations == {}


def test_analysis_state_serializes_as_stable_structured_data() -> None:
    evidence = Evidence(
        id="evidence-1",
        type=EvidenceType.DIFF,
        source="src/task_service.py:142",
        claim="The changed identifier matches the duplicate-key field in the log.",
        strength=EvidenceStrength.STRONG,
    )
    owner = OwnerCandidate(
        identity="team-storage",
        display_name="Storage Team",
        reason="The change is in the storage-owned identifier generator.",
    )
    investigation = InvestigationResult(
        error_id="error-1",
        hypothesis="The identifier change caused key collisions.",
        root_cause="The identifier generator no longer produces unique values.",
        owner_candidate=owner,
        evidence=[evidence],
        confidence=Confidence.HIGH,
    )
    validation = ValidationResult(
        error_id="error-1",
        verdict=ValidationVerdict.ACCEPTED,
        final_confidence=Confidence.HIGH,
        root_cause_valid=True,
        ownership_valid=True,
        confidence_valid=True,
        accepted_evidence_ids=["evidence-1"],
    )
    state = AnalysisState(
        stage=AnalysisStage.DONE,
        build=BuildContext(
            build_id="build-1",
            job_name="unit-tests",
            repository="example/ci-owner-agent",
            log_ref="jenkins://unit-tests/build-1/console",
            diff_ref="git://example/ci-owner-agent/build-1.diff",
        ),
        discoveries=[
            ErrorCandidate(
                id="error-1",
                title="Duplicate key",
                primary_message="E11000 duplicate key error",
                log_spans=[LogSpan(start_line=42, end_line=43)],
            )
        ],
        investigations={"error-1": investigation},
        validations={"error-1": validation},
        final_report=FinalReport(
            build_id="build-1",
            summary="One validated build error.",
            errors=[
                FinalError(
                    error_id="error-1",
                    title="Duplicate key",
                    root_cause=investigation.root_cause,
                    owner=owner.identity,
                    confidence=Confidence.HIGH,
                    evidence=[evidence],
                    validation_status=ValidationVerdict.ACCEPTED,
                )
            ],
        ),
    )

    python_data = state.model_dump()
    json_data = json.loads(state.model_dump_json())

    assert python_data["stage"] is AnalysisStage.DONE
    assert json_data["stage"] == "done"
    assert json_data["investigations"]["error-1"]["evidence"][0]["type"] == "diff"
    assert json_data["final_report"]["errors"][0]["owner"] == "team-storage"
