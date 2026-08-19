from pathlib import Path

import pytest

from ci_owner_agent.discovery import (
    ChunkDiscovery,
    ErrorFragment,
    DiscoveryInvariantError,
    DiscoveryModelError,
    JsonlTraceSink,
    LLMDiscoverer,
    LocalLogReader,
    LogChunk,
    OpenAICompatibleDiscoveryModel,
    TraceEvent,
)
from ci_owner_agent.domain import (
    AnalysisStage,
    BuildContext,
    Confidence,
    ErrorCandidate,
    InvestigationResult,
    ValidationResult,
    ValidationVerdict,
)
from ci_owner_agent.orchestrator import AnalysisOrchestrator


def write_log(tmp_path: Path, content: str) -> Path:
    log_path = tmp_path / "build.log"
    log_path.write_text(content, encoding="utf-8")
    return log_path


class FakeDiscoveryModel:
    model_name = "fake-discovery-model"

    def __init__(self, results: dict[str, ChunkDiscovery]) -> None:
        self.results = results
        self.seen_chunks: list[LogChunk] = []

    def extract_errors(self, chunk: LogChunk) -> ChunkDiscovery:
        self.seen_chunks.append(chunk)
        return self.results.get(chunk.chunk_id, ChunkDiscovery())


class MalformedDiscoveryModel:
    model_name = "malformed-model"

    def extract_errors(self, chunk: LogChunk) -> ChunkDiscovery:
        return {"errors": [{"title": "missing required fields"}]}  # type: ignore[return-value]


class RecordingTraceSink:
    def __init__(self) -> None:
        self.events: list[TraceEvent] = []

    def emit(self, event: TraceEvent) -> None:
        self.events.append(event)


class FailingTraceSink:
    def emit(self, event: TraceEvent) -> None:
        raise OSError("trace disk unavailable")


class FakeJsonTransport:
    def __init__(self, response: dict[str, object]) -> None:
        self.response = response
        self.requests: list[dict[str, object]] = []

    def post_json(
        self,
        url: str,
        headers: dict[str, str],
        payload: dict[str, object],
        timeout: float,
    ) -> dict[str, object]:
        self.requests.append(
            {
                "url": url,
                "headers": headers,
                "payload": payload,
                "timeout": timeout,
            }
        )
        return self.response


class FixedInvestigator:
    def investigate(
        self, build: BuildContext, error: ErrorCandidate
    ) -> InvestigationResult:
        return InvestigationResult(
            error_id=error.id,
            hypothesis="Fixed test investigation",
            confidence=Confidence.MEDIUM,
        )


class FixedValidator:
    def validate(
        self, build: BuildContext, investigation: InvestigationResult
    ) -> ValidationResult:
        return ValidationResult(
            error_id=investigation.error_id,
            verdict=ValidationVerdict.ACCEPTED,
            final_confidence=Confidence.MEDIUM,
            root_cause_valid=True,
            ownership_valid=True,
            confidence_valid=True,
        )


def make_build(log_path: Path) -> BuildContext:
    return BuildContext(
        build_id="build-1",
        job_name="unit-tests",
        log_ref=str(log_path),
    )


def test_local_log_reader_numbers_every_line_in_a_small_log(tmp_path: Path) -> None:
    log_path = write_log(tmp_path, "first\nsecond\nthird\n")
    reader = LocalLogReader(chunk_lines=10, overlap_lines=2)

    chunks = list(reader.iter_chunks(str(log_path)))

    assert len(chunks) == 1
    assert chunks[0].chunk_id == "chunk-0001"
    assert chunks[0].start_line == 1
    assert chunks[0].end_line == 3
    assert chunks[0].text == "[1] first\n[2] second\n[3] third"


def test_local_log_reader_overlaps_chunks_without_missing_the_tail(
    tmp_path: Path,
) -> None:
    log_path = write_log(
        tmp_path, "\n".join(f"line-{number}" for number in range(1, 9)) + "\n"
    )
    reader = LocalLogReader(chunk_lines=4, overlap_lines=1)

    chunks = list(reader.iter_chunks(str(log_path)))

    assert [(chunk.start_line, chunk.end_line) for chunk in chunks] == [
        (1, 4),
        (4, 7),
        (7, 8),
    ]
    covered_lines = {
        line_number
        for chunk in chunks
        for line_number in range(chunk.start_line, chunk.end_line + 1)
    }
    assert covered_lines == set(range(1, 9))
    assert chunks[-1].text == "[7] line-7\n[8] line-8"


def test_local_log_reader_yields_no_chunks_for_an_empty_log(tmp_path: Path) -> None:
    log_path = write_log(tmp_path, "")

    chunks = list(LocalLogReader().iter_chunks(str(log_path)))

    assert chunks == []


def test_llm_discoverer_scans_every_chunk_and_returns_no_false_errors(
    tmp_path: Path,
) -> None:
    log_path = write_log(tmp_path, "one\ntwo\nthree\nfour\nfive\n")
    model = FakeDiscoveryModel({})
    discoverer = LLMDiscoverer(
        reader=LocalLogReader(chunk_lines=3, overlap_lines=1),
        model=model,
    )

    candidates = discoverer.discover(make_build(log_path))

    assert candidates == []
    assert [chunk.chunk_id for chunk in model.seen_chunks] == [
        "chunk-0001",
        "chunk-0002",
    ]


def test_llm_discoverer_builds_stable_candidate_from_original_log_lines(
    tmp_path: Path,
) -> None:
    log_path = write_log(
        tmp_path,
        "setup\nAssertionError: expected 1 to equal 2\n    at test.py:9\nsummary\n",
    )
    model = FakeDiscoveryModel(
        {
            "chunk-0001": ChunkDiscovery(
                errors=[
                    ErrorFragment(
                        title="Assertion failed",
                        category="assertion",
                        primary_message="expected 1 to equal 2",
                        test_name="calculates total",
                        start_line=2,
                        end_line=3,
                    )
                ]
            )
        }
    )
    discoverer = LLMDiscoverer(LocalLogReader(), model)

    candidates = discoverer.discover(make_build(log_path))

    assert len(candidates) == 1
    assert candidates[0].id == "error-001"
    assert candidates[0].title == "Assertion failed"
    assert candidates[0].category == "assertion"
    assert candidates[0].test_name == "calculates total"
    assert candidates[0].log_spans[0].start_line == 2
    assert candidates[0].log_spans[0].end_line == 3
    assert candidates[0].log_spans[0].excerpt == (
        "AssertionError: expected 1 to equal 2\n    at test.py:9"
    )


def test_llm_discoverer_sorts_multiple_errors_before_assigning_ids(
    tmp_path: Path,
) -> None:
    log_path = write_log(tmp_path, "early\nmiddle\nlate\n")
    model = FakeDiscoveryModel(
        {
            "chunk-0001": ChunkDiscovery(
                errors=[
                    ErrorFragment(
                        title="Late error",
                        primary_message="late",
                        start_line=3,
                        end_line=3,
                    ),
                    ErrorFragment(
                        title="Early error",
                        primary_message="early",
                        start_line=1,
                        end_line=1,
                    ),
                ]
            )
        }
    )

    candidates = LLMDiscoverer(LocalLogReader(), model).discover(
        make_build(log_path)
    )

    assert [(candidate.id, candidate.title) for candidate in candidates] == [
        ("error-001", "Early error"),
        ("error-002", "Late error"),
    ]


def test_llm_discoverer_rejects_fragment_outside_its_chunk(tmp_path: Path) -> None:
    log_path = write_log(tmp_path, "one\ntwo\nthree\n")
    model = FakeDiscoveryModel(
        {
            "chunk-0001": ChunkDiscovery(
                errors=[
                    ErrorFragment(
                        title="Hallucinated span",
                        primary_message="outside",
                        start_line=10,
                        end_line=11,
                    )
                ]
            )
        }
    )
    trace = RecordingTraceSink()

    with pytest.raises(DiscoveryInvariantError, match="outside chunk"):
        LLMDiscoverer(LocalLogReader(), model, trace_sink=trace).discover(
            make_build(log_path)
        )

    assert [event.event_type for event in trace.events][-2:] == [
        "fragment_rejected",
        "discovery_failed",
    ]


def test_llm_discoverer_rejects_malformed_structured_output(tmp_path: Path) -> None:
    log_path = write_log(tmp_path, "one\n")
    trace = RecordingTraceSink()

    with pytest.raises(DiscoveryInvariantError, match="structured output"):
        LLMDiscoverer(
            LocalLogReader(),
            MalformedDiscoveryModel(),
            trace_sink=trace,
        ).discover(make_build(log_path))

    assert [event.event_type for event in trace.events][-2:] == [
        "fragment_rejected",
        "discovery_failed",
    ]


def test_llm_discoverer_merges_only_overlapping_duplicate_fragments(
    tmp_path: Path,
) -> None:
    log_path = write_log(
        tmp_path,
        "start\nTimeout elsewhere\nmiddle\nTimeout at boundary\nafter\nTimeout elsewhere\n",
    )
    boundary_fragment = ErrorFragment(
        title="Timeout",
        category="timeout",
        primary_message="operation timed out",
        start_line=4,
        end_line=4,
    )
    model = FakeDiscoveryModel(
        {
            "chunk-0001": ChunkDiscovery(errors=[boundary_fragment]),
            "chunk-0002": ChunkDiscovery(
                errors=[
                    boundary_fragment,
                    ErrorFragment(
                        title="Timeout",
                        category="timeout",
                        primary_message="operation timed out",
                        start_line=6,
                        end_line=6,
                    ),
                ]
            ),
        }
    )
    discoverer = LLMDiscoverer(
        LocalLogReader(chunk_lines=4, overlap_lines=1),
        model,
        trace_sink=(trace := RecordingTraceSink()),
    )

    candidates = discoverer.discover(make_build(log_path))

    assert [(candidate.id, candidate.log_spans[0].start_line) for candidate in candidates] == [
        ("error-001", 4),
        ("error-002", 6),
    ]
    assert [event.event_type for event in trace.events].count("fragment_merged") == 1


def test_discovery_trace_records_pipeline_without_raw_content_by_default(
    tmp_path: Path,
) -> None:
    log_path = write_log(tmp_path, "secret-token\nAssertionError\n")
    model = FakeDiscoveryModel(
        {
            "chunk-0001": ChunkDiscovery(
                errors=[
                    ErrorFragment(
                        title="Assertion",
                        primary_message="assertion failed",
                        start_line=2,
                        end_line=2,
                    )
                ]
            )
        }
    )
    trace = RecordingTraceSink()
    discoverer = LLMDiscoverer(LocalLogReader(), model, trace_sink=trace)

    discoverer.discover(make_build(log_path))

    event_types = [event.event_type for event in trace.events]
    assert event_types == [
        "discovery_started",
        "log_loaded",
        "chunk_created",
        "llm_request",
        "llm_response",
        "fragment_accepted",
        "candidate_created",
        "discovery_completed",
    ]
    assert {event.run_id for event in trace.events} == {
        trace.events[0].run_id
    }
    request = next(event for event in trace.events if event.event_type == "llm_request")
    assert request.data["prompt_version"] == "discover-v1"
    assert request.data["model"] == "fake-discovery-model"
    serialized_trace = "\n".join(event.model_dump_json() for event in trace.events)
    assert "secret-token" not in serialized_trace


def test_jsonl_trace_sink_writes_one_json_event_per_line(tmp_path: Path) -> None:
    trace_path = tmp_path / "traces" / "discover.jsonl"
    sink = JsonlTraceSink(trace_path)
    event = TraceEvent(
        run_id="run-1",
        event_type="discovery_started",
        build_id="build-1",
    )

    sink.emit(event)
    sink.emit(event.model_copy(update={"event_type": "discovery_completed"}))

    lines = trace_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert '"event_type":"discovery_started"' in lines[0]
    assert '"event_type":"discovery_completed"' in lines[1]


def test_trace_failure_does_not_change_discovery_result(tmp_path: Path) -> None:
    log_path = write_log(tmp_path, "AssertionError\n")
    model = FakeDiscoveryModel(
        {
            "chunk-0001": ChunkDiscovery(
                errors=[
                    ErrorFragment(
                        title="Assertion",
                        primary_message="assertion failed",
                        start_line=1,
                        end_line=1,
                    )
                ]
            )
        }
    )

    candidates = LLMDiscoverer(
        LocalLogReader(), model, trace_sink=FailingTraceSink()
    ).discover(make_build(log_path))

    assert [candidate.id for candidate in candidates] == ["error-001"]


def test_trace_captures_chunk_content_only_when_explicitly_enabled(
    tmp_path: Path,
) -> None:
    log_path = write_log(tmp_path, "local-secret-content\n")
    trace = RecordingTraceSink()

    LLMDiscoverer(
        LocalLogReader(),
        FakeDiscoveryModel({}),
        trace_sink=trace,
        capture_content=True,
    ).discover(make_build(log_path))

    serialized_trace = "\n".join(event.model_dump_json() for event in trace.events)
    assert "local-secret-content" in serialized_trace


def test_openai_compatible_model_sends_prompt_and_parses_structured_result() -> None:
    transport = FakeJsonTransport(
        {
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"errors":[{"title":"Assertion failed",'
                            '"category":"assertion",'
                            '"primary_message":"expected 1",'
                            '"test_name":null,"start_line":12,"end_line":13}]}'
                        )
                    }
                }
            ],
            "usage": {"prompt_tokens": 100, "completion_tokens": 25},
        }
    )
    model = OpenAICompatibleDiscoveryModel(
        base_url="https://llm.example/v1",
        api_key="test-key",
        model="test-model",
        transport=transport,
    )
    chunk = LogChunk(
        chunk_id="chunk-0001",
        start_line=11,
        end_line=14,
        text="[11] setup\n[12] AssertionError\n[13] at test\n[14] summary",
    )

    result = model.extract_errors(chunk)

    assert result.errors[0].start_line == 12
    request = transport.requests[0]
    assert request["url"] == "https://llm.example/v1/chat/completions"
    assert request["headers"] == {
        "Authorization": "Bearer test-key",
        "Content-Type": "application/json",
    }
    payload = request["payload"]
    assert isinstance(payload, dict)
    assert payload["model"] == "test-model"
    assert payload["response_format"] == {"type": "json_object"}
    assert "[12] AssertionError" in str(payload["messages"])
    assert "禁止判断根因" in str(payload["messages"])
    assert model.last_metadata["input_tokens"] == 100
    assert model.last_metadata["output_tokens"] == 25


def test_openai_compatible_model_rejects_invalid_provider_json() -> None:
    transport = FakeJsonTransport(
        {
            "choices": [
                {"message": {"content": '{"errors":[{"title":"bad"}]}'}}
            ]
        }
    )
    model = OpenAICompatibleDiscoveryModel(
        base_url="https://llm.example/v1",
        api_key="test-key",
        model="test-model",
        transport=transport,
    )

    with pytest.raises(DiscoveryModelError, match="invalid response"):
        model.extract_errors(
            LogChunk(
                chunk_id="chunk-0001",
                start_line=1,
                end_line=1,
                text="[1] failure",
            )
        )


def test_llm_discoverer_satisfies_existing_orchestrator_contract(
    tmp_path: Path,
) -> None:
    log_path = write_log(tmp_path, "AssertionError\n")
    discoverer = LLMDiscoverer(
        LocalLogReader(),
        FakeDiscoveryModel(
            {
                "chunk-0001": ChunkDiscovery(
                    errors=[
                        ErrorFragment(
                            title="Assertion",
                            primary_message="assertion failed",
                            start_line=1,
                            end_line=1,
                        )
                    ]
                )
            }
        ),
    )
    orchestrator = AnalysisOrchestrator(
        discoverer=discoverer,
        investigator=FixedInvestigator(),
        validator=FixedValidator(),
    )

    state = orchestrator.run(make_build(log_path))

    assert state.stage is AnalysisStage.AGGREGATING
    assert [error.id for error in state.discoveries] == ["error-001"]
