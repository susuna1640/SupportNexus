"""Optional Langfuse observability for the SupportNexus runtime.

The application must remain usable when Langfuse is not configured.  All
helpers in this module therefore degrade to no-ops when the SDK or credentials
are unavailable.  Langfuse is imported lazily so ``load_dotenv()`` has already
run before the SDK reads its environment variables.
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional

logger = logging.getLogger(__name__)

_client: Any = None
_client_initialized = False
_warned_sdk_missing = False


def _truthy(name: str, default: str = "true") -> bool:
    return os.getenv(name, default).strip().lower() not in {"0", "false", "no", "off"}


def enabled() -> bool:
    """Return whether Langfuse tracing is configured and explicitly enabled."""
    return (
        _truthy("LANGFUSE_TRACING_ENABLED")
        and bool(os.getenv("LANGFUSE_PUBLIC_KEY", "").strip())
        and bool(os.getenv("LANGFUSE_SECRET_KEY", "").strip())
    )


def client() -> Any:
    """Return the process-wide Langfuse client, or ``None`` when disabled."""
    global _client, _client_initialized, _warned_sdk_missing

    if not enabled():
        return None
    if _client_initialized:
        return _client

    _client_initialized = True
    try:
        # Import only after dotenv has loaded LANGFUSE_* variables.
        from langfuse import get_client

        _client = get_client()
    except ImportError:
        if not _warned_sdk_missing:
            logger.warning("Langfuse tracing 已配置，但 Python SDK 未安装；跳过埋点")
            _warned_sdk_missing = True
    except Exception as ex:
        logger.warning("Langfuse 初始化失败，跳过埋点: %s", ex)
    return _client


def capture(value: Any) -> Any:
    """Capture useful IO by default, with a single switch for sensitive data.

    Only explicit user/model/tool payloads are passed to this helper.  API
    keys and client configuration are never included in observations.
    """
    if _truthy("LANGFUSE_CAPTURE_CONTENT"):
        return value
    if isinstance(value, dict):
        return {"captured": False, "field_count": len(value)}
    if isinstance(value, list):
        return {"captured": False, "item_count": len(value)}
    return "[内容采集已关闭]"


@contextmanager
def observation(
    name: str,
    *,
    as_type: str = "span",
    **kwargs: Any,
) -> Iterator[Optional[Any]]:
    """Create a current Langfuse observation without making it a hard dependency."""
    lf = client()
    if lf is None:
        yield None
        return

    try:
        manager = lf.start_as_current_observation(as_type=as_type, name=name, **kwargs)
    except Exception as ex:
        logger.warning("Langfuse observation 创建失败 (%s): %s", name, ex)
        yield None
        return

    # Do not swallow exceptions raised by the application body.  The SDK itself
    # handles export failures; application failures should still reach callers.
    with manager as obs:
        yield obs


@contextmanager
def trace_attributes(**kwargs: Any) -> Iterator[None]:
    """Propagate request-scoped user/session metadata to all child observations."""
    lf = client()
    if lf is None:
        yield
        return

    try:
        from langfuse import propagate_attributes

        manager = propagate_attributes(**kwargs)
    except Exception as ex:
        logger.warning("Langfuse trace 属性设置失败: %s", ex)
        yield
        return

    with manager:
        yield


def update(obs: Any, **kwargs: Any) -> None:
    """Best-effort observation update."""
    if obs is None:
        return
    try:
        obs.update(**kwargs)
    except Exception as ex:
        logger.debug("Langfuse observation 更新失败: %s", ex)


def usage_details(response: Any) -> Dict[str, int]:
    """Extract Anthropic-compatible token usage for Langfuse cost analytics."""
    usage = getattr(response, "usage", None)
    if usage is None and isinstance(response, dict):
        usage = response.get("usage")
    if usage is None:
        return {}

    def read(name: str) -> Optional[int]:
        value = getattr(usage, name, None)
        if value is None and isinstance(usage, dict):
            value = usage.get(name)
        try:
            return int(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    details: Dict[str, int] = {}
    input_tokens = read("input_tokens")
    output_tokens = read("output_tokens")
    if input_tokens is not None:
        details["input_tokens"] = input_tokens
    if output_tokens is not None:
        details["output_tokens"] = output_tokens
    return details


def shutdown() -> None:
    """Flush buffered events during normal application shutdown."""
    lf = client()
    if lf is None:
        return
    try:
        lf.shutdown()
    except Exception as ex:
        logger.warning("Langfuse 关闭/刷新失败: %s", ex)
