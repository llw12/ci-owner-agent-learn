"""Workflow control plane for CI error analysis."""

from copy import deepcopy
from typing import Protocol

from ci_owner_agent.domain import (
    AnalysisStage,
    AnalysisState,
    BuildContext,
    Confidence,
    ErrorCandidate,
    InvestigationResult,
    ValidationResult,
)


_CONFIDENCE_ORDER = (
    Confidence.NONE,
    Confidence.LOW,
    Confidence.MEDIUM,
    Confidence.HIGH,
)


class InvalidStateTransition(Exception):
    """Raised when a workflow stage change is not allowed."""


class WorkflowInvariantError(Exception):
    """Raised when an agent result cannot be correlated with its input."""


ALLOWED_TRANSITIONS: dict[AnalysisStage, frozenset[AnalysisStage]] = {
    AnalysisStage.INIT: frozenset(
        {AnalysisStage.DISCOVERING, AnalysisStage.FAILED}
    ),
    AnalysisStage.DISCOVERING: frozenset(
        {AnalysisStage.INVESTIGATING, AnalysisStage.FAILED}
    ),
    AnalysisStage.INVESTIGATING: frozenset(
        {AnalysisStage.VALIDATING, AnalysisStage.FAILED}
    ),
    AnalysisStage.VALIDATING: frozenset(
        {AnalysisStage.AGGREGATING, AnalysisStage.FAILED}
    ),
    AnalysisStage.AGGREGATING: frozenset({AnalysisStage.FAILED}),
    AnalysisStage.DONE: frozenset(),
    AnalysisStage.FAILED: frozenset(),
}


def transition(
    state: AnalysisState,
    target: AnalysisStage,
    **changes: object,
) -> AnalysisState:
    """Return a fully validated snapshot at the requested next stage."""

    if target not in ALLOWED_TRANSITIONS[state.stage]:
        raise InvalidStateTransition(
            f"cannot transition from {state.stage.value} to {target.value}"
        )

    snapshot_data = state.model_dump()
    snapshot_data.update(deepcopy(changes))
    snapshot_data["stage"] = target
    return AnalysisState.model_validate(snapshot_data)


class Discoverer(Protocol):
    def discover(self, build: BuildContext) -> list[ErrorCandidate]: ...


class Investigator(Protocol):
    def investigate(
        self, build: BuildContext, error: ErrorCandidate
    ) -> InvestigationResult: ...


class Validator(Protocol):
    def validate(
        self, build: BuildContext, investigation: InvestigationResult
    ) -> ValidationResult: ...


class AnalysisOrchestrator:
    """Run the analysis workflow through validation using injected capabilities."""

    def __init__(
        self,
        discoverer: Discoverer,
        investigator: Investigator,
        validator: Validator,
    ) -> None:
        self.discoverer = discoverer
        self.investigator = investigator
        self.validator = validator

    def run(self, build: BuildContext) -> AnalysisState:
        state = transition(
            AnalysisState(build=build),
            AnalysisStage.DISCOVERING,
        )

        try:
            discoveries = self.discoverer.discover(
                state.build.model_copy(deep=True)
            )
            state = transition(
                state,
                AnalysisStage.INVESTIGATING,
                discoveries=discoveries,
            )

            investigations: dict[str, InvestigationResult] = {}
            for error in state.discoveries:
                expected_error_id = error.id
                investigation = self.investigator.investigate(
                    state.build.model_copy(deep=True),
                    error.model_copy(deep=True),
                )
                if investigation.error_id != expected_error_id:
                    raise WorkflowInvariantError(
                        f"investigator returned {investigation.error_id!r} for "
                        f"error {expected_error_id!r}"
                    )
                investigations[expected_error_id] = investigation
            state = transition(
                state,
                AnalysisStage.VALIDATING,
                investigations=investigations,
            )

            validations: dict[str, ValidationResult] = {}
            for investigation in state.investigations.values():
                expected_error_id = investigation.error_id
                maximum_confidence = investigation.confidence
                validation = self.validator.validate(
                    state.build.model_copy(deep=True),
                    investigation.model_copy(deep=True),
                )
                if validation.error_id != expected_error_id:
                    raise WorkflowInvariantError(
                        f"validator returned {validation.error_id!r} for "
                        f"investigation {expected_error_id!r}"
                    )
                if _CONFIDENCE_ORDER.index(
                    validation.final_confidence
                ) > _CONFIDENCE_ORDER.index(maximum_confidence):
                    raise WorkflowInvariantError(
                        "validator confidence cannot exceed investigation confidence "
                        f"for {expected_error_id!r}"
                    )
                validations[expected_error_id] = validation
            return transition(
                state,
                AnalysisStage.AGGREGATING,
                validations=validations,
            )
        except WorkflowInvariantError:
            raise
        except Exception as error:
            return transition(
                state,
                AnalysisStage.FAILED,
                fatal_error=f"{type(error).__name__}: {error}",
            )
