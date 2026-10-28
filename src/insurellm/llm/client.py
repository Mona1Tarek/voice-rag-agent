from __future__ import annotations
import os, time
from dataclasses import dataclass, asdict
from typing import Any
from dotenv import load_dotenv
from openai import OpenAI

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
    """Client for OpenAI-compatible provider APIs (e.g. Groq/OpenRouter)."""
    def __init__(self) -> None:
        self.provider = os.getenv("LLM_PROVIDER")
        self.model = os.getenv("LLM_MODEL")
        
        key = os.getenv("LLM_API_KEY") 
        if not key:
            raise ValueError("Set LLM_API_KEY in .env")
        
        self.client = OpenAI(
            api_key=key,
            base_url=os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1"),
            timeout=float(os.getenv("LLM_TIMEOUT_SECONDS", "60")),
            max_retries=int(os.getenv("LLM_MAX_RETRIES", "0")),
        )
        
        self.temperature = float(os.getenv("LLM_TEMPERATURE", "0.2"))
        self.max_tokens = int(os.getenv("LLM_MAX_OUTPUT_TOKENS", "500"))
        self.stream_usage = os.getenv("LLM_STREAM_USAGE", "false").lower() == "true"


    def generate(self, messages: list[dict[str, str]], *, stream: bool = True) -> LLMResult:
        start = time.perf_counter()
        first_token = None
        chunks: list[str] = []
        usage = None
        finish = None
        headers: dict[str, str] = {}
        status = 200
        
        try:
            # dictionary containing the arguments that will be sent to the model
            params: dict[str, Any] = dict(
                model=self.model, 
                messages=messages, 
                temperature=self.temperature,
                max_tokens=self.max_tokens, 
                stream=stream,
            )
            
            if stream and self.stream_usage:
                params["stream_options"] = {"include_usage": True}
                
            raw = self.client.chat.completions.with_raw_response.create(**params)
            
            headers = _headers(raw.headers)
            response = raw.parse()
            
            if stream:
                for event in response:
                    now = time.perf_counter()
                    if getattr(event, "usage", None):
                        usage = event.usage
                    for choice in event.choices:
                        if choice.delta and choice.delta.content:
                            first_token = first_token or now
                            chunks.append(choice.delta.content)
                        if choice.finish_reason:
                            finish = choice.finish_reason
                end = time.perf_counter()
                text = "".join(chunks)
                ttft = (first_token - start) * 1000 if first_token else None
                gen_ms = (end - first_token) * 1000 if first_token else None
            else:
                end = time.perf_counter()
                text = response.choices[0].message.content or ""
                usage = response.usage
                finish = response.choices[0].finish_reason
                ttft, gen_ms = None, None
                
            return LLMResult(
                text, 
                self.provider, 
                self.model, 
                (end-start)*1000, 
                ttft, 
                gen_ms,
                _usage(usage, "prompt_tokens"), 
                _usage(usage, "completion_tokens"),
                _usage(usage, "total_tokens"), 
                finish, status, headers,
            )
            
        except Exception as exc:
            response = getattr(exc, "response", None)
            status = getattr(response, "status_code", None)
            
            return LLMResult(
                "", self.provider, self.model, (time.perf_counter()-start)*1000,
                (first_token-start)*1000 if first_token else None, None,
                None, None, None, None, status, _headers(getattr(response, "headers", {})),
                f"{type(exc).__name__}: {exc}",
            )

def _usage(obj: Any, field: str) -> int | None:
    value = getattr(obj, field, None) if obj else None
    return int(value) if value is not None else None

def _headers(headers: Any) -> dict[str, str]:
    wanted = ("x-ratelimit-limit-requests", "x-ratelimit-remaining-requests",
              "x-ratelimit-reset-requests", "x-ratelimit-limit-tokens",
              "x-ratelimit-remaining-tokens", "x-ratelimit-reset-tokens",
              "retry-after", "x-request-id")
    values = {str(k).lower(): str(v) for k, v in (headers or {}).items()}
    return {k: values[k] for k in wanted if k in values}
