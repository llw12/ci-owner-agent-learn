import pytest
from pydantic import ValidationError

from ci_owner_agent.domain import (
    AnalysisStage,
    AnalysisState,
    BuildContext,
    Confidence,
    ErrorCandidate,
    InvestigationResult,
    ValidationResult,
    ValidationVerdict,
)
from ci_owner_agent.orchestrator import (
    AnalysisOrchestrator,
    InvalidStateTransition,
    WorkflowInvariantError,
    transition,
)


def make_build() -> BuildContext:
    return BuildContext(
        build_id="build-1",
        job_name="unit-tests",
        log_ref="jenkins://unit-tests/build-1/console",
    )


class FakeDiscoverer:
    def __init__(
        self, discoveries: list[ErrorCandidate], events: list[str] | None = None
    ) -> None:
        self.discoveries = discoveries
        self.events = events

    def discover(self, build: BuildContext) -> list[ErrorCandidate]:
        if self.events is not None:
            self.events.append("discover")
        return self.discoveries


class FakeInvestigator:
    def __init__(self, events: list[str] | None = None) -> None:
        self.events = events

    def investigate(
        self, build: BuildContext, error: ErrorCandidate
    ) -> InvestigationResult:
        if self.events is not None:
            self.events.append(f"investigate:{error.id}")
        return InvestigationResult(
            error_id=error.id,
            hypothesis=f"Hypothesis for {error.id}",
            confidence=Confidence.MEDIUM,
        )


class FakeValidator:
    def __init__(self, events: list[str] | None = None) -> None:
        self.events = events

    def validate(
        self, build: BuildContext, investigation: InvestigationResult
    ) -> ValidationResult:
        if self.events is not None:
            self.events.append(f"validate:{investigation.error_id}")
        return ValidationResult(
            error_id=investigation.error_id,
            verdict=ValidationVerdict.ACCEPTED,
            final_confidence=investigation.confidence,
            root_cause_valid=True,
            ownership_valid=True,
            confidence_valid=True,
        )


class MismatchedInvestigator:
    def investigate(
        self, build: BuildContext, error: ErrorCandidate
    ) -> InvestigationResult:
        return InvestigationResult(
            error_id="error-999",
            hypothesis="This result belongs to another error.",
            confidence=Confidence.MEDIUM,
        )


class MutatingMismatchedInvestigator:
    def investigate(
        self, build: BuildContext, error: ErrorCandidate
    ) -> InvestigationResult:
        error.id = "error-999"
        return InvestigationResult(
            error_id=error.id,
            hypothesis="The input ID was changed before returning.",
            confidence=Confidence.MEDIUM,
        )


class MismatchedValidator:
    def validate(
        self, build: BuildContext, investigation: InvestigationResult
    ) -> ValidationResult:
        return ValidationResult(
            error_id="error-999",
            verdict=ValidationVerdict.REJECTED,
            final_confidence=Confidence.NONE,
            root_cause_valid=False,
            ownership_valid=False,
            confidence_valid=False,
        )


class UpgradingValidator:
    def validate(
        self, build: BuildContext, investigation: InvestigationResult
    ) -> ValidationResult:
        return ValidationResult(
            error_id=investigation.error_id,
            verdict=ValidationVerdict.ACCEPTED,
            final_confidence=Confidence.HIGH,
            root_cause_valid=True,
            ownership_valid=True,
            confidence_valid=False,
        )


class MutatingUpgradingValidator:
    def validate(
        self, build: BuildContext, investigation: InvestigationResult
    ) -> ValidationResult:
        investigation.confidence = Confidence.HIGH
        return ValidationResult(
            error_id=investigation.error_id,
            verdict=ValidationVerdict.ACCEPTED,
            final_confidence=Confidence.HIGH,
            root_cause_valid=True,
            ownership_valid=True,
            confidence_valid=False,
        )


class FailingDiscoverer:
    def discover(self, build: BuildContext) -> list[ErrorCandidate]:
        raise RuntimeError("boom")


class FailingInvestigator:
    def investigate(
        self, build: BuildContext, error: ErrorCandidate
    ) -> InvestigationResult:
        raise RuntimeError("investigation failed")


class FailingValidator:
    def validate(
        self, build: BuildContext, investigation: InvestigationResult
    ) -> ValidationResult:
        raise RuntimeError("validation failed")


def test_transition_returns_a_new_validated_snapshot() -> None:
    initial = AnalysisState(build=make_build())

    discovering = transition(initial, AnalysisStage.DISCOVERING)

    assert initial.stage is AnalysisStage.INIT
    assert discovering is not initial
    assert discovering.stage is AnalysisStage.DISCOVERING


@pytest.mark.parametrize(
    ("source", "target"),
    [
        (AnalysisStage.INIT, AnalysisStage.VALIDATING),
        (AnalysisStage.DONE, AnalysisStage.DISCOVERING),
        (AnalysisStage.FAILED, AnalysisStage.DISCOVERING),
    ],
)
def test_transition_rejects_illegal_and_terminal_stage_changes(
    source: AnalysisStage, target: AnalysisStage
) -> None:
    state = AnalysisState(stage=source, build=make_build())

    with pytest.raises(InvalidStateTransition):
        transition(state, target)


def test_transition_revalidates_changed_state_data() -> None:
    investigation = InvestigationResult(
        error_id="error-1",
        hypothesis="The change may cause the failure.",
        confidence=Confidence.MEDIUM,
    )
    state = AnalysisState(
        stage=AnalysisStage.VALIDATING,
        build=make_build(),
        investigations={"error-1": investigation},
    )
    upgraded_validation = ValidationResult(
        error_id="error-1",
        verdict=ValidationVerdict.ACCEPTED,
        final_confidence=Confidence.HIGH,
        root_cause_valid=True,
        ownership_valid=True,
        confidence_valid=False,
    )

    with pytest.raises(ValidationError, match="cannot exceed"):
        transition(
            state,
            AnalysisStage.AGGREGATING,
            validations={"error-1": upgraded_validation},
        )


def test_orchestrator_runs_the_happy_path_to_aggregating() -> None:
    discoveries = [
        ErrorCandidate(id="error-1", title="First", primary_message="first"),
        ErrorCandidate(id="error-2", title="Second", primary_message="second"),
    ]
    orchestrator = AnalysisOrchestrator(
        discoverer=FakeDiscoverer(discoveries),
        investigator=FakeInvestigator(),
        validator=FakeValidator(),
    )

    state = orchestrator.run(make_build())

    assert state.stage is AnalysisStage.AGGREGATING
    assert [error.id for error in state.discoveries] == ["error-1", "error-2"]
    assert set(state.investigations) == {"error-1", "error-2"}
    assert set(state.validations) == {"error-1", "error-2"}
    assert state.final_report is None


def test_orchestrator_calls_agents_in_serial_stage_order() -> None:
    events: list[str] = []
    discoveries = [
        ErrorCandidate(id="error-1", title="First", primary_message="first"),
        ErrorCandidate(id="error-2", title="Second", primary_message="second"),
    ]
    orchestrator = AnalysisOrchestrator(
        discoverer=FakeDiscoverer(discoveries, events),
        investigator=FakeInvestigator(events),
        validator=FakeValidator(events),
    )

    orchestrator.run(make_build())

    assert events == [
        "discover",
        "investigate:error-1",
        "investigate:error-2",
        "validate:error-1",
        "validate:error-2",
    ]


def test_orchestrator_allows_empty_discoveries() -> None:
    orchestrator = AnalysisOrchestrator(
        discoverer=FakeDiscoverer([]),
        investigator=FakeInvestigator(),
        validator=FakeValidator(),
    )

    state = orchestrator.run(make_build())

    assert state.stage is AnalysisStage.AGGREGATING
    assert state.discoveries == []
    assert state.investigations == {}
    assert state.validations == {}


def test_orchestrator_rejects_mismatched_investigation_error_id() -> None:
    error = ErrorCandidate(
        id="error-1", title="First", primary_message="first failure"
    )
    orchestrator = AnalysisOrchestrator(
        discoverer=FakeDiscoverer([error]),
        investigator=MismatchedInvestigator(),
        validator=FakeValidator(),
    )

    with pytest.raises(WorkflowInvariantError, match="error-999"):
        orchestrator.run(make_build())


def test_agent_cannot_mutate_snapshot_to_hide_investigation_id_mismatch() -> None:
    error = ErrorCandidate(
        id="error-1", title="First", primary_message="first failure"
    )
    orchestrator = AnalysisOrchestrator(
        discoverer=FakeDiscoverer([error]),
        investigator=MutatingMismatchedInvestigator(),
        validator=FakeValidator(),
    )

    with pytest.raises(WorkflowInvariantError, match="error-999"):
        orchestrator.run(make_build())


def test_orchestrator_rejects_mismatched_validation_error_id() -> None:
    error = ErrorCandidate(
        id="error-1", title="First", primary_message="first failure"
    )
    orchestrator = AnalysisOrchestrator(
        discoverer=FakeDiscoverer([error]),
        investigator=FakeInvestigator(),
        validator=MismatchedValidator(),
    )

    with pytest.raises(WorkflowInvariantError, match="error-999"):
        orchestrator.run(make_build())


def test_orchestrator_rejects_runtime_confidence_upgrade() -> None:
    error = ErrorCandidate(
        id="error-1", title="First", primary_message="first failure"
    )
    orchestrator = AnalysisOrchestrator(
        discoverer=FakeDiscoverer([error]),
        investigator=FakeInvestigator(),
        validator=UpgradingValidator(),
    )

    with pytest.raises(WorkflowInvariantError, match="confidence"):
        orchestrator.run(make_build())


def test_agent_cannot_mutate_snapshot_to_hide_confidence_upgrade() -> None:
    error = ErrorCandidate(
        id="error-1", title="First", primary_message="first failure"
    )
    orchestrator = AnalysisOrchestrator(
        discoverer=FakeDiscoverer([error]),
        investigator=FakeInvestigator(),
        validator=MutatingUpgradingValidator(),
    )

    with pytest.raises(WorkflowInvariantError, match="confidence"):
        orchestrator.run(make_build())


def test_orchestrator_records_discoverer_failure() -> None:
    orchestrator = AnalysisOrchestrator(
        discoverer=FailingDiscoverer(),
        investigator=FakeInvestigator(),
        validator=FakeValidator(),
    )

    state = orchestrator.run(make_build())

    assert state.stage is AnalysisStage.FAILED
    assert state.fatal_error == "RuntimeError: boom"


def test_orchestrator_records_investigator_failure() -> None:
    error = ErrorCandidate(
        id="error-1", title="First", primary_message="first failure"
    )
    orchestrator = AnalysisOrchestrator(
        discoverer=FakeDiscoverer([error]),
        investigator=FailingInvestigator(),
        validator=FakeValidator(),
    )

    state = orchestrator.run(make_build())

    assert state.stage is AnalysisStage.FAILED
    assert state.fatal_error == "RuntimeError: investigation failed"


def test_orchestrator_records_validator_failure() -> None:
    error = ErrorCandidate(
        id="error-1", title="First", primary_message="first failure"
    )
    orchestrator = AnalysisOrchestrator(
        discoverer=FakeDiscoverer([error]),
        investigator=FakeInvestigator(),
        validator=FailingValidator(),
    )

    state = orchestrator.run(make_build())

    assert state.stage is AnalysisStage.FAILED
    assert state.fatal_error == "RuntimeError: validation failed"
