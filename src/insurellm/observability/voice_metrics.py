
from __future__ import annotations

import logging
from typing import Any

from insurellm.observability.tracing import record_observation

logger = logging.getLogger("insurellm.voice.metrics")


def _serialize_metrics(metrics: Any) -> dict:
    """Convert a LiveKit metrics object into JSON-compatible data."""
    if hasattr(metrics, "model_dump"):
        return metrics.model_dump(mode="json", exclude_none=True)

    if hasattr(metrics, "dict"):
        return metrics.dict(exclude_none=True)

    return {
        key: value
        for key, value in vars(metrics).items()
        if not key.startswith("_")
    }


def _handle_metrics(
    component: str,
    model: str,
    room_name: str,
    metrics: Any,
) -> None:
    """Log and record one component metrics event."""
    data = _serialize_metrics(metrics)

    # Avoid recording empty or uninformative events.
    if not data:
        return

    event_type = data.get("type", type(metrics).__name__)

    logger.info(
        "[VOICE METRICS] component=%s model=%s metrics=%s",
        component,
        model,
        data,
    )

    try:
        record_observation(
            name=f"voice.{component}.{event_type}",
            input={
                "room_name": room_name,
                "component": component,
                "model": model,
                "request_id": data.get("request_id"),
            },
            output=data,
            metadata={
                "service.name": "insurellm-voice-agent",
                "voice.component": component,
                "voice.model": model,
                "voice.room_name": room_name,
                "voice.metrics_type": event_type,
            },
        )
    except Exception:
        # Observability failures must not interrupt the voice session.
        logger.exception(
            "Failed to record %s metrics in Langfuse", component
        )


def attach_component_metrics(
    component_instance: Any,
    *,
    component: str,
    model: str,
    room_name: str,
) -> None:
    """Subscribe to metrics emitted by an STT, LLM, or TTS instance."""

    def on_metrics(metrics: Any) -> None:
        _handle_metrics(
            component=component,
            model=model,
            room_name=room_name,
            metrics=metrics,
        )

    component_instance.on("metrics_collected", on_metrics)
