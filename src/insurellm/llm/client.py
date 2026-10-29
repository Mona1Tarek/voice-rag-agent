
from __future__ import annotations

import os
import time
from dataclasses import dataclass, asdict
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from insurellm.observability.tracing import observation

load_dotenv()


@dataclass
class LLMResult:
    text: str
    provider: str
    model: str
    latency_ms: float
    ttft_ms: float | None
    generation_duration_ms: float | None
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    finish_reason: str | None
    status_code: int | None = 200
    rate_limit_headers: dict[str, str] | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def tokens_per_second(self) -> float | None:
        if self.output_tokens is None or not self.generation_duration_ms:
            return None

        return self.output_tokens / (self.generation_duration_ms / 1000)


class LLMClient:
    """Client for OpenAI-compatible provider APIs."""

    def __init__(self) -> None:
        self.provider = os.getenv("LLM_PROVIDER")
        self.model = os.getenv("LLM_MODEL")

        key = os.getenv("LLM_API_KEY")
        if not key:
            raise ValueError("Set LLM_API_KEY in .env")

        if not self.model:
            raise ValueError("Set LLM_MODEL in .env")

        self.client = OpenAI(
            api_key=key,
            base_url=os.getenv(
                "LLM_BASE_URL",
                "https://api.groq.com/openai/v1",
            ),
            timeout=float(os.getenv("LLM_TIMEOUT_SECONDS", "60")),
            max_retries=int(os.getenv("LLM_MAX_RETRIES", "0")),
        )

        self.temperature = float(os.getenv("LLM_TEMPERATURE", "0.2"))
        self.max_tokens = int(
            os.getenv("LLM_MAX_OUTPUT_TOKENS", "500")
        )
        self.stream_usage = (
            os.getenv("LLM_STREAM_USAGE", "false").lower() == "true"
        )

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        stream: bool = True,
    ) -> LLMResult:
        start = time.perf_counter()
        first_token = None
        chunks: list[str] = []
        usage = None
        finish = None
        headers: dict[str, str] = {}
        status = 200

        # Parameters useful for debugging and comparing model runs.
        model_parameters = {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": stream,
        }

        with observation(
            as_type="generation",
            name="llm-generation",
            input=messages,
            metadata={
                "provider": self.provider,
                "model": self.model,
                "model_parameters": model_parameters,
            },
        ) as span:
            try:
                params: dict[str, Any] = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                    "stream": stream,
                }

                if stream and self.stream_usage:
                    params["stream_options"] = {
                        "include_usage": True,
                    }

                raw = (
                    self.client.chat.completions
                    .with_raw_response.create(**params)
                )

                headers = _headers(raw.headers)
                response = raw.parse()

                if stream:
                    for event in response:
                        now = time.perf_counter()

                        if getattr(event, "usage", None):
                            usage = event.usage

                        for choice in event.choices:
                            if choice.delta and choice.delta.content:
                                if first_token is None:
                                    first_token = now

                                chunks.append(choice.delta.content)

                            if choice.finish_reason:
                                finish = choice.finish_reason

                    end = time.perf_counter()
                    text = "".join(chunks)

                    ttft = (
                        (first_token - start) * 1000
                        if first_token is not None
                        else None
                    )
                    generation_ms = (
                        (end - first_token) * 1000
                        if first_token is not None
                        else None
                    )

                else:
                    end = time.perf_counter()
                    text = response.choices[0].message.content or ""
                    usage = response.usage
                    finish = response.choices[0].finish_reason

                    ttft = None
                    generation_ms = None

                result = LLMResult(
                    text=text,
                    provider=self.provider,
                    model=self.model,
                    latency_ms=(end - start) * 1000,
                    ttft_ms=ttft,
                    generation_duration_ms=generation_ms,
                    input_tokens=_usage(usage, "prompt_tokens"),
                    output_tokens=_usage(usage, "completion_tokens"),
                    total_tokens=_usage(usage, "total_tokens"),
                    finish_reason=finish,
                    status_code=status,
                    rate_limit_headers=headers,
                )

                # Record the completed generation in Langfuse.
                span.update(
                    output=result.text,
                    usage_details={
                        "input": result.input_tokens,
                        "output": result.output_tokens,
                        "total": result.total_tokens,
                    },
                    metadata={
                        "latency_ms": result.latency_ms,
                        "ttft_ms": result.ttft_ms,
                        "generation_duration_ms": (
                            result.generation_duration_ms
                        ),
                        "tokens_per_second": result.tokens_per_second,
                        "finish_reason": result.finish_reason,
                        "status_code": result.status_code,
                        "rate_limit_headers": result.rate_limit_headers,
                    },
                )

                return result

            except Exception as exc:
                response = getattr(exc, "response", None)
                status = getattr(response, "status_code", None)
                error_headers = _headers(
                    getattr(response, "headers", {})
                )

                result = LLMResult(
                    text="",
                    provider=self.provider,
                    model=self.model,
                    latency_ms=(time.perf_counter() - start) * 1000,
                    ttft_ms=(
                        (first_token - start) * 1000
                        if first_token is not None
                        else None
                    ),
                    generation_duration_ms=None,
                    input_tokens=None,
                    output_tokens=None,
                    total_tokens=None,
                    finish_reason=None,
                    status_code=status,
                    rate_limit_headers=error_headers,
                    error=f"{type(exc).__name__}: {exc}",
                )

                span.update(
                    level="ERROR",
                    status_message=result.error,
                    output={
                        "error": result.error,
                        "status_code": result.status_code,
                        "latency_ms": result.latency_ms,
                    },
                    metadata={
                        "rate_limit_headers": error_headers,
                    },
                )

                return result


def _usage(obj: Any, field: str) -> int | None:
    value = getattr(obj, field, None) if obj else None
    return int(value) if value is not None else None


def _headers(headers: Any) -> dict[str, str]:
    wanted = (
        "x-ratelimit-limit-requests",
        "x-ratelimit-remaining-requests",
        "x-ratelimit-reset-requests",
        "x-ratelimit-limit-tokens",
        "x-ratelimit-remaining-tokens",
        "x-ratelimit-reset-tokens",
        "retry-after",
        "x-request-id",
    )

    values = {
        str(key).lower(): str(value)
        for key, value in (headers or {}).items()
    }

    return {
        key: values[key]
        for key in wanted
        if key in values
    }
