"""
config/logging_config.py
========================
Structured JSON logging configuration for all system components.
"""

import logging
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from config.settings import app as app_cfg


class JSONFormatter(logging.Formatter):
    """Formats log records as single-line JSON for log aggregation."""

    RESERVED_ATTRS = {
        "args", "asctime", "created", "exc_info", "exc_text",
        "filename", "funcName", "levelname", "levelno", "lineno",
        "message", "module", "msecs", "msg", "name", "pathname",
        "process", "processName", "relativeCreated", "stack_info",
        "thread", "threadName",
    }

    def format(self, record: logging.LogRecord) -> str:  # noqa: A003
        log_dict = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "message": record.getMessage(),
            "environment": app_cfg.environment,
        }

        # Attach exception info
        if record.exc_info:
            log_dict["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]),
                "traceback": traceback.format_exception(*record.exc_info),
            }

        # Attach any extra fields passed to the logger
        for key, value in record.__dict__.items():
            if key not in self.RESERVED_ATTRS and not key.startswith("_"):
                log_dict[key] = value

        return json.dumps(log_dict, default=str, ensure_ascii=False)


class StructuredLogger:
    """
    Thin wrapper around :class:`logging.Logger` that accepts keyword arguments
    on every log-level method and routes them through ``extra`` so that
    :class:`JSONFormatter` can attach them to the emitted JSON record.

    Usage::

        logger = get_logger(__name__)
        logger.info("Connected", host="localhost", port=5432)
    """

    def __init__(self, inner: logging.Logger) -> None:
        self._inner = inner

    # ------------------------------------------------------------------ #
    # Delegate attribute access (e.g. .name, .level, .handlers) to inner  #
    # ------------------------------------------------------------------ #
    def __getattr__(self, item: str):
        return getattr(self._inner, item)

    # ------------------------------------------------------------------ #
    # Log-level helpers                                                    #
    # ------------------------------------------------------------------ #

    def _log(self, level: int, msg: str, *args, **kwargs) -> None:
        extra = kwargs.pop("extra", {})
        extra.update(kwargs)          # absorb structlog-style key=value pairs
        exc_info = kwargs.pop("exc_info", False) if "exc_info" in extra else False
        self._inner.log(level, msg, *args, extra=extra, exc_info=exc_info)

    def debug(self, msg: str, *args, **kwargs) -> None:
        self._log(logging.DEBUG, msg, *args, **kwargs)

    def info(self, msg: str, *args, **kwargs) -> None:
        self._log(logging.INFO, msg, *args, **kwargs)

    def warning(self, msg: str, *args, **kwargs) -> None:
        self._log(logging.WARNING, msg, *args, **kwargs)

    def error(self, msg: str, *args, **kwargs) -> None:
        self._log(logging.ERROR, msg, *args, **kwargs)

    def critical(self, msg: str, *args, **kwargs) -> None:
        self._log(logging.CRITICAL, msg, *args, **kwargs)

    def exception(self, msg: str, *args, **kwargs) -> None:
        kwargs["exc_info"] = True
        self._log(logging.ERROR, msg, *args, **kwargs)

    def isEnabledFor(self, level: int) -> bool:
        return self._inner.isEnabledFor(level)


def get_logger(
    name: str,
    log_file: Optional[str] = None,
    level: Optional[str] = None,
) -> StructuredLogger:
    """
    Return a :class:`StructuredLogger` with JSON formatting to stdout and
    optionally to a rotating file.

    Args:
        name:     Logger name, typically ``__name__`` of the calling module.
        log_file: Optional absolute path to a log file.
        level:    Override the default log level from ``AppConfig``.
    """
    inner = logging.getLogger(name)

    # Avoid adding duplicate handlers in Lambda re-use / hot-reload scenarios
    if not inner.handlers:
        effective_level = getattr(
            logging, (level or app_cfg.log_level).upper(), logging.INFO
        )
        inner.setLevel(effective_level)

        formatter = JSONFormatter()

        # stdout handler
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setFormatter(formatter)
        inner.addHandler(stdout_handler)

        # optional file handler
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_path, encoding="utf-8")
            file_handler.setFormatter(formatter)
            inner.addHandler(file_handler)

        inner.propagate = False

    return StructuredLogger(inner)
