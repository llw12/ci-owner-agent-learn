"""Domain contracts shared by the future analysis stages."""

from enum import Enum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class DomainModel(BaseModel):
    """Strict base model for all domain contracts."""

    model_config = ConfigDict(extra="forbid")


class EvidenceType(str, Enum):
    LOG = "log"
    DIFF = "diff"
    CODE = "code"
    SYMBOL = "symbol"
    HISTORY = "history"


class EvidenceStrength(str, Enum):
    STRONG = "strong"
    SUPPORTING = "supporting"
    WEAK = "weak"


class Confidence(str, Enum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ValidationVerdict(str, Enum):
    ACCEPTED = "accepted"
    DOWNGRADED = "downgraded"
    REJECTED = "rejected"


class AnalysisStage(str, Enum):
    INIT = "init"
    DISCOVERING = "discovering"
    INVESTIGATING = "investigating"
    VALIDATING = "validating"
    AGGREGATING = "aggregating"
    DONE = "done"
    FAILED = "failed"


class BuildContext(DomainModel):
    build_id: str
    job_name: str
    repository: str | None = None
    branch: str | None = None
    commit_sha: str | None = None
    build_url: str | None = None
    log_ref: str
    diff_ref: str | None = None


class LogSpan(DomainModel):
    """A one-based inclusive span in a build log."""

    start_line: int = Field(ge=1)
    end_line: int
    excerpt: str | None = None

    @model_validator(mode="after")
    def end_must_not_precede_start(self) -> Self:
        if self.end_line < self.start_line:
            raise ValueError("end_line must be greater than or equal to start_line")
        return self


class ErrorCandidate(DomainModel):
    """An independently investigable error observed in build logs."""

    id: str
    title: str
    category: str | None = None
    primary_message: str
    test_name: str | None = None
    log_spans: list[LogSpan] = Field(default_factory=list)


class Evidence(DomainModel):
    id: str
    type: EvidenceType
    source: str
    claim: str
    strength: EvidenceStrength
    detail: str | None = None


class OwnerCandidate(DomainModel):
    identity: str
    display_name: str | None = None
    reason: str


class InvestigationResult(DomainModel):
    error_id: str
    hypothesis: str
    root_cause: str | None = None
    owner_candidate: OwnerCandidate | None = None
    evidence: list[Evidence] = Field(default_factory=list)
    counter_evidence: list[Evidence] = Field(default_factory=list)
    confidence: Confidence = Confidence.NONE
    unresolved_questions: list[str] = Field(default_factory=list)


class ValidationResult(DomainModel):
    error_id: str
    verdict: ValidationVerdict
    original_confidence: Confidence
    final_confidence: Confidence
    owner_candidate: OwnerCandidate | None = None
    root_cause_valid: bool
    ownership_valid: bool
    confidence_valid: bool
    accepted_evidence_ids: list[str] = Field(default_factory=list)
    rejected_evidence_ids: list[str] = Field(default_factory=list)
    validation_reasons: list[str] = Field(default_factory=list)
    needs_reinvestigation: bool = False

    @model_validator(mode="after")
    def final_confidence_cannot_exceed_original(self) -> Self:
        confidence_order = (
            Confidence.NONE,
            Confidence.LOW,
            Confidence.MEDIUM,
            Confidence.HIGH,
        )
        if confidence_order.index(self.final_confidence) > confidence_order.index(
            self.original_confidence
        ):
            raise ValueError("final_confidence cannot exceed original_confidence")
        return self


class FinalError(DomainModel):
    error_id: str
    title: str
    root_cause: str | None = None
    owner: str | None = None
    confidence: Confidence
    evidence: list[Evidence] = Field(default_factory=list)
    validation_status: ValidationVerdict


class FinalReport(DomainModel):
    build_id: str
    summary: str
    errors: list[FinalError] = Field(default_factory=list)


class AnalysisState(DomainModel):
    stage: AnalysisStage = AnalysisStage.INIT
    build: BuildContext
    discoveries: list[ErrorCandidate] = Field(default_factory=list)
    investigations: dict[str, InvestigationResult] = Field(default_factory=dict)
    validations: dict[str, ValidationResult] = Field(default_factory=dict)
    final_report: FinalReport | None = None
    fatal_error: str | None = None
