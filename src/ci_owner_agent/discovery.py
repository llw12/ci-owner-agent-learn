"""Deterministic local-log discovery pipeline."""

from collections.abc import Iterable
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
from time import perf_counter
from typing import Protocol
from urllib.request import Request, urlopen
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ci_owner_agent.domain import BuildContext, ErrorCandidate, LogSpan


class DiscoveryModelBase(BaseModel):
    model_config = ConfigDict(extra="forbid")


class LogChunk(DiscoveryModelBase):
    chunk_id: str
    start_line: int
    end_line: int
    text: str


class ErrorFragment(DiscoveryModelBase):
    title: str
    category: str | None = None
    primary_message: str
    test_name: str | None = None
    start_line: int
    end_line: int


class ChunkDiscovery(DiscoveryModelBase):
    errors: list[ErrorFragment] = Field(default_factory=list)


class DiscoveryModel(Protocol):
    model_name: str

    def extract_errors(self, chunk: LogChunk) -> ChunkDiscovery: ...


DISCOVERY_PROMPT_VERSION = "discover-v1"

DISCOVERY_SYSTEM_PROMPT = """你是 CI 日志错误发现器。

你的唯一任务是从提供的日志片段中提取所有值得独立调查的错误事实，只能依据日志内容。

禁止判断根因、责任人、代码修改、root/secondary 关系或多个错误之间的因果关系。禁止使用日志中不存在的信息。

对每个错误输出 title、category、primary_message、test_name、start_line、end_line。test_name 仅在日志明确存在时填写，否则为 null。category 只能描述日志表象类型，不确定时为 null。

不要仅因为出现 ERROR、failed、exception 等单词就认定构建失败；测试可能故意触发并成功验证异常。不要因为 message 相同就合并不同位置的失败。构建系统对同一测试失败的 process exit、make error、build failure 等包装不应机械地产生额外错误。

如果当前片段没有值得独立调查的错误，返回空 errors。只返回符合给定 JSON schema 的 JSON，不要输出推理过程。"""


class JsonTransport(Protocol):
    def post_json(
        self,
        url: str,
        headers: dict[str, str],
        payload: dict[str, object],
        timeout: float,
    ) -> object: ...


class UrllibJsonTransport:
    def post_json(
        self,
        url: str,
        headers: dict[str, str],
        payload: dict[str, object],
        timeout: float,
    ) -> object:
        request = Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))


class DiscoveryModelError(Exception):
    """Raised when an external discovery model request or response fails."""

    def __init__(
        self,
        message: str,
        *,
        reason: str,
        metadata: dict[str, object] | None = None,
        raw_content: str | None = None,
    ) -> None:
        super().__init__(message)
        self.reason = reason
        self.metadata = dict(metadata or {})
        self.raw_content = raw_content


class OpenAICompatibleDiscoveryModel:
    """Call an OpenAI-compatible chat-completions JSON endpoint."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float = 60.0,
        transport: JsonTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model_name = model
        self.timeout = timeout
        self.transport = transport or UrllibJsonTransport()
        self.last_metadata: dict[str, object] = {}

    @staticmethod
    def _request_metadata(
        started: float,
        metadata: dict[str, object] | None = None,
    ) -> dict[str, object]:
        return {
            **(metadata or {}),
            "latency_ms": round((perf_counter() - started) * 1000, 3),
        }

    @classmethod
    def from_env(cls) -> "OpenAICompatibleDiscoveryModel":
        names = {
            "base_url": "CI_OWNER_LLM_BASE_URL",
            "api_key": "CI_OWNER_LLM_API_KEY",
            "model": "CI_OWNER_LLM_MODEL",
        }
        values = {key: os.getenv(name) for key, name in names.items()}
        missing = [names[key] for key, value in values.items() if not value]
        if missing:
            raise DiscoveryModelError(
                "missing model configuration: " + ", ".join(missing),
                reason="configuration_error",
            )
        return cls(
            base_url=str(values["base_url"]),
            api_key=str(values["api_key"]),
            model=str(values["model"]),
        )

    def extract_errors(self, chunk: LogChunk) -> ChunkDiscovery:
        self.last_metadata = {}
        schema = json.dumps(
            ChunkDiscovery.model_json_schema(), ensure_ascii=False
        )
        payload: dict[str, object] = {
            "model": self.model_name,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [
                {
                    "role": "system",
                    "content": f"{DISCOVERY_SYSTEM_PROMPT}\n\nJSON schema:\n{schema}",
                },
                {"role": "user", "content": chunk.text},
            ],
        }
        started = perf_counter()
        try:
            response = self.transport.post_json(
                f"{self.base_url}/chat/completions",
                {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                payload,
                self.timeout,
            )
        except json.JSONDecodeError as error:
            self.last_metadata = self._request_metadata(started)
            raise DiscoveryModelError(
                f"model {self.model_name!r} returned an invalid provider response",
                reason="invalid_provider_response",
                metadata=self.last_metadata,
            ) from error
        except Exception as error:
            self.last_metadata = self._request_metadata(started)
            raise DiscoveryModelError(
                f"model {self.model_name!r} request failed",
                reason="transport_error",
                metadata=self.last_metadata,
            ) from error

        response_metadata: dict[str, object] = {}
        try:
            if not isinstance(response, dict):
                raise TypeError("provider response must be a JSON object")
            usage = response.get("usage")
            if isinstance(usage, dict):
                prompt_tokens = usage.get("prompt_tokens")
                completion_tokens = usage.get("completion_tokens")
                if isinstance(prompt_tokens, int):
                    response_metadata["input_tokens"] = prompt_tokens
                if isinstance(completion_tokens, int):
                    response_metadata["output_tokens"] = completion_tokens
            choices = response["choices"]
            if not isinstance(choices, list) or not choices:
                raise TypeError("provider response has no choices")
            choice = choices[0]
            if not isinstance(choice, dict):
                raise TypeError("provider choice must be an object")
            message = choice["message"]
            if not isinstance(message, dict):
                raise TypeError("provider message must be an object")
            content = message["content"]
            if not isinstance(content, str):
                raise TypeError("provider message content must be text")
        except (KeyError, TypeError) as error:
            self.last_metadata = self._request_metadata(
                started, response_metadata
            )
            raise DiscoveryModelError(
                f"model {self.model_name!r} returned an invalid provider response",
                reason="invalid_provider_response",
                metadata=self.last_metadata,
            ) from error

        try:
            result = ChunkDiscovery.model_validate_json(content)
        except ValidationError as error:
            self.last_metadata = self._request_metadata(
                started,
                {
                    **response_metadata,
                    "content_sha256": sha256(
                        content.encode("utf-8")
                    ).hexdigest(),
                    "content_length": len(content),
                    "validation_error_count": error.error_count(),
                },
            )
            raise DiscoveryModelError(
                f"model {self.model_name!r} returned invalid structured output",
                reason="invalid_structured_output",
                metadata=self.last_metadata,
                raw_content=content,
            ) from error

        self.last_metadata = self._request_metadata(started, response_metadata)
        return result


class TraceEvent(DiscoveryModelBase):
    run_id: str
    event_type: str
    build_id: str
    chunk_id: str | None = None
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    data: dict[str, object] = Field(default_factory=dict)


class TraceSink(Protocol):
    def emit(self, event: TraceEvent) -> None: ...


class NoOpTraceSink:
    def emit(self, event: TraceEvent) -> None:
        return None


class JsonlTraceSink:
    """Append trace events to an explicitly configured local JSONL file."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def emit(self, event: TraceEvent) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as trace_file:
            trace_file.write(event.model_dump_json())
            trace_file.write("\n")


class DiscoveryInvariantError(Exception):
    """Raised when structured model output violates a pipeline invariant."""


def _messages_match(left: str, right: str) -> bool:
    return " ".join(left.split()).casefold() == " ".join(right.split()).casefold()


def _spans_highly_overlap(left: ErrorFragment, right: ErrorFragment) -> bool:
    overlap = max(
        0,
        min(left.end_line, right.end_line)
        - max(left.start_line, right.start_line)
        + 1,
    )
    shorter_span = min(
        left.end_line - left.start_line + 1,
        right.end_line - right.start_line + 1,
    )
    return overlap / shorter_span >= 0.5


def _deduplicate_overlaps(
    fragments: list[ErrorFragment],
) -> tuple[list[ErrorFragment], list[dict[str, object]]]:
    deduplicated: list[ErrorFragment] = []
    merges: list[dict[str, object]] = []
    for fragment in fragments:
        for index, existing in enumerate(deduplicated):
            if _messages_match(
                existing.primary_message, fragment.primary_message
            ) and _spans_highly_overlap(existing, fragment):
                merged = ErrorFragment(
                    title=existing.title,
                    category=existing.category,
                    primary_message=existing.primary_message,
                    test_name=existing.test_name or fragment.test_name,
                    start_line=min(existing.start_line, fragment.start_line),
                    end_line=max(existing.end_line, fragment.end_line),
                )
                deduplicated[index] = merged
                merges.append(
                    {
                        "kept_span": [existing.start_line, existing.end_line],
                        "merged_span": [fragment.start_line, fragment.end_line],
                        "result_span": [merged.start_line, merged.end_line],
                        "primary_message": merged.primary_message,
                    }
                )
                break
        else:
            deduplicated.append(fragment)
    return deduplicated, merges


class LocalLogReader:
    """Read a local log as deterministic, overlapping, line-numbered chunks."""

    def __init__(self, chunk_lines: int = 400, overlap_lines: int = 40) -> None:
        if chunk_lines < 1:
            raise ValueError("chunk_lines must be at least 1")
        if overlap_lines < 0 or overlap_lines >= chunk_lines:
            raise ValueError(
                "overlap_lines must be non-negative and less than chunk_lines"
            )
        self.chunk_lines = chunk_lines
        self.overlap_lines = overlap_lines

    def iter_chunks(self, log_ref: str) -> Iterable[LogChunk]:
        lines = Path(log_ref).read_text(encoding="utf-8", errors="replace").splitlines()
        start_index = 0
        chunk_number = 1

        while start_index < len(lines):
            end_index = min(start_index + self.chunk_lines, len(lines))
            numbered_text = "\n".join(
                f"[{line_number}] {lines[line_number - 1]}"
                for line_number in range(start_index + 1, end_index + 1)
            )
            yield LogChunk(
                chunk_id=f"chunk-{chunk_number:04d}",
                start_line=start_index + 1,
                end_line=end_index,
                text=numbered_text,
            )
            if end_index == len(lines):
                break
            start_index = end_index - self.overlap_lines
            chunk_number += 1

    def read_span(self, log_ref: str, start_line: int, end_line: int) -> str:
        lines = Path(log_ref).read_text(encoding="utf-8", errors="replace").splitlines()
        return "\n".join(lines[start_line - 1 : end_line])


class LLMDiscoverer:
    """Scan every deterministic chunk and convert model fragments to candidates."""

    def __init__(
        self,
        reader: LocalLogReader,
        model: DiscoveryModel,
        trace_sink: TraceSink | None = None,
        capture_content: bool = False,
    ) -> None:
        self.reader = reader
        self.model = model
        self.trace_sink = trace_sink or NoOpTraceSink()
        self.capture_content = capture_content

    def _emit(self, event: TraceEvent) -> None:
        try:
            self.trace_sink.emit(event)
        except Exception:
            pass

    def discover(self, build: BuildContext) -> list[ErrorCandidate]:
        run_id = uuid4().hex
        self._emit(
            TraceEvent(
                run_id=run_id,
                event_type="discovery_started",
                build_id=build.build_id,
                data={"log_ref": build.log_ref},
            )
        )
        try:
            return self._discover(build, run_id)
        except Exception as error:
            self._emit(
                TraceEvent(
                    run_id=run_id,
                    event_type="discovery_failed",
                    build_id=build.build_id,
                    data={
                        "error_type": type(error).__name__,
                        "error": str(error),
                    },
                )
            )
            raise

    def _discover(
        self, build: BuildContext, run_id: str
    ) -> list[ErrorCandidate]:
        chunks = list(self.reader.iter_chunks(build.log_ref))
        self._emit(
            TraceEvent(
                run_id=run_id,
                event_type="log_loaded",
                build_id=build.build_id,
                data={
                    "chunk_count": len(chunks),
                    "sha256": sha256(Path(build.log_ref).read_bytes()).hexdigest(),
                },
            )
        )
        fragments: list[ErrorFragment] = []
        for chunk in chunks:
            chunk_data: dict[str, object] = {
                "start_line": chunk.start_line,
                "end_line": chunk.end_line,
                "sha256": sha256(chunk.text.encode("utf-8")).hexdigest(),
            }
            if self.capture_content:
                chunk_data["content"] = chunk.text
            self._emit(
                TraceEvent(
                    run_id=run_id,
                    event_type="chunk_created",
                    build_id=build.build_id,
                    chunk_id=chunk.chunk_id,
                    data=chunk_data,
                )
            )
            request_data: dict[str, object] = {
                "prompt_version": DISCOVERY_PROMPT_VERSION,
                "model": self.model.model_name,
                "start_line": chunk.start_line,
                "end_line": chunk.end_line,
            }
            if self.capture_content:
                request_data["content"] = chunk.text
            self._emit(
                TraceEvent(
                    run_id=run_id,
                    event_type="llm_request",
                    build_id=build.build_id,
                    chunk_id=chunk.chunk_id,
                    data=request_data,
                )
            )
            try:
                raw_result = self.model.extract_errors(chunk)
            except DiscoveryModelError as error:
                failure_data = {
                    **error.metadata,
                    "reason": error.reason,
                    "model": self.model.model_name,
                    "prompt_version": DISCOVERY_PROMPT_VERSION,
                }
                if self.capture_content and error.raw_content is not None:
                    failure_data["raw_content"] = error.raw_content
                self._emit(
                    TraceEvent(
                        run_id=run_id,
                        event_type="llm_failed",
                        build_id=build.build_id,
                        chunk_id=chunk.chunk_id,
                        data=failure_data,
                    )
                )
                raise
            result_data = (
                raw_result.model_dump()
                if isinstance(raw_result, BaseModel)
                else raw_result
            )
            try:
                discovery = ChunkDiscovery.model_validate(result_data)
            except ValidationError as error:
                self._emit(
                    TraceEvent(
                        run_id=run_id,
                        event_type="fragment_rejected",
                        build_id=build.build_id,
                        chunk_id=chunk.chunk_id,
                        data={
                            "reason": "invalid_structured_output",
                            "validation_errors": error.error_count(),
                        },
                    )
                )
                raise DiscoveryInvariantError(
                    f"invalid structured output for {chunk.chunk_id}"
                ) from error
            response_data: dict[str, object] = {
                "errors": discovery.model_dump(mode="json")["errors"]
            }
            model_metadata = getattr(self.model, "last_metadata", None)
            if isinstance(model_metadata, dict):
                response_data.update(model_metadata)
            self._emit(
                TraceEvent(
                    run_id=run_id,
                    event_type="llm_response",
                    build_id=build.build_id,
                    chunk_id=chunk.chunk_id,
                    data=response_data,
                )
            )
            for fragment in discovery.errors:
                if not (
                    chunk.start_line
                    <= fragment.start_line
                    <= fragment.end_line
                    <= chunk.end_line
                ):
                    self._emit(
                        TraceEvent(
                            run_id=run_id,
                            event_type="fragment_rejected",
                            build_id=build.build_id,
                            chunk_id=chunk.chunk_id,
                            data={
                                "reason": "span_outside_chunk",
                                "fragment": fragment.model_dump(mode="json"),
                            },
                        )
                    )
                    raise DiscoveryInvariantError(
                        f"fragment span {fragment.start_line}-{fragment.end_line} "
                        f"is outside chunk {chunk.start_line}-{chunk.end_line}"
                    )
                fragments.append(fragment)
                self._emit(
                    TraceEvent(
                        run_id=run_id,
                        event_type="fragment_accepted",
                        build_id=build.build_id,
                        chunk_id=chunk.chunk_id,
                        data=fragment.model_dump(mode="json"),
                    )
                )

        fragments.sort(
            key=lambda fragment: (
                fragment.start_line,
                fragment.end_line,
                fragment.primary_message,
            )
        )
        fragments, merges = _deduplicate_overlaps(fragments)
        for merge in merges:
            self._emit(
                TraceEvent(
                    run_id=run_id,
                    event_type="fragment_merged",
                    build_id=build.build_id,
                    data=merge,
                )
            )
        candidates: list[ErrorCandidate] = []
        for index, fragment in enumerate(fragments, start=1):
            candidate = ErrorCandidate(
                id=f"error-{index:03d}",
                title=fragment.title,
                category=fragment.category,
                primary_message=fragment.primary_message,
                test_name=fragment.test_name,
                log_spans=[
                    LogSpan(
                        start_line=fragment.start_line,
                        end_line=fragment.end_line,
                        excerpt=self.reader.read_span(
                            build.log_ref,
                            fragment.start_line,
                            fragment.end_line,
                        ),
                    )
                ],
            )
            candidates.append(candidate)
            self._emit(
                TraceEvent(
                    run_id=run_id,
                    event_type="candidate_created",
                    build_id=build.build_id,
                    data={
                        "error_id": candidate.id,
                        "start_line": fragment.start_line,
                        "end_line": fragment.end_line,
                    },
                )
            )
        self._emit(
            TraceEvent(
                run_id=run_id,
                event_type="discovery_completed",
                build_id=build.build_id,
                data={"candidate_count": len(candidates)},
            )
        )
        return candidates
