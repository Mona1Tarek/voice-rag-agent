
import os
from contextlib import contextmanager
from typing import Any

from opentelemetry.sdk.trace import TracerProvider  
from livekit.agents.telemetry import set_tracer_provider   
from langfuse import Langfuse, get_client


def setup_langfuse(metadata=None):
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    base_url = os.getenv("LANGFUSE_BASE_URL")

    if not public_key or not secret_key or not base_url:
        raise ValueError("Missing Langfuse configuration")

    # responsible for providing the tracing infrastructure that we will use to create and record spans
    provider = TracerProvider()
    # livekit can use this Tracer Provider in the telemetry system
    # This allows us to integrate the tracing for the Voice Agent with the same observability system we are already using
    set_tracer_provider(provider, metadata=metadata)

    Langfuse(
        public_key=public_key,
        secret_key=secret_key,
        base_url=base_url,
        tracer_provider=provider,
        should_export_span=lambda span: True,
    )
    return provider


# It helps us manage resources 
# We use it around code whose execution we want to monitor, we start recording before the operation begins and end it once the operation is complete
@contextmanager
def observation(
    *,
    name: str,
    as_type: str = "span",
    input: Any = None,
    metadata: dict | None = None,
    **kwargs,
):
    langfuse = get_client()

    # creating the observation
    with langfuse.start_as_current_observation(
        as_type=as_type,
        name=name,
        input=input,
        metadata=metadata,
        **kwargs,
    ) as span:
        try:
            yield span
        except Exception as exc:
            span.update(level="ERROR", status_message=str(exc))
            raise


# We use it to record an event we already have the data (completed), for example: metrics received from LiveKit
def record_observation(
    *,
    name: str,
    input: Any = None,
    output: Any = None,
    metadata: dict | None = None,
    level: str = "DEFAULT",
    status_message: str | None = None,
) -> None:
    """Record a completed event as a standalone Langfuse span."""
    langfuse = get_client()

    # create the observation of type span
    with langfuse.start_as_current_observation(
        name=name,
        as_type="span",
        input=input,
        metadata=metadata,
    ) as span:
        update = {"level": level}
        if output is not None:
            update["output"] = output
        if status_message is not None:
            update["status_message"] = status_message
        span.update(**update)
