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


def get_logger(
    name: str,
    log_file: Optional[str] = None,
    level: Optional[str] = None,
) -> logging.Logger:
    """
    Return a named logger with JSON formatting to stdout and optionally a file.

    Args:
        name:     Logger name, typically __name__ of the calling module.
        log_file: Optional path to a rotating log file.
        level:    Override the default log level from settings.
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers in Lambda re-use scenarios
    if logger.handlers:
        return logger

    effective_level = getattr(logging, (level or app_cfg.log_level).upper(), logging.INFO)
    logger.setLevel(effective_level)

    formatter = JSONFormatter()

    # --- stdout handler ---
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)

    # --- file handler (optional) ---
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.propagate = False
    return logger
