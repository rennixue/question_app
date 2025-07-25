from logging import Filter, LogRecord
from typing import Any

from .context import request_id


class RequestIdFilter(Filter):
    def __init__(self, name: str = "", max_len: int | None = None) -> None:
        super().__init__(name)
        self._max_len = max_len

    def filter(self, record: LogRecord) -> bool:
        s = request_id.get()
        if self._max_len:
            s = s[: self._max_len]
        record.request_id = s
        return True


DEV_LOG_CFG: dict[str, Any] = {
    "version": 1,
    "disable_existing_loggers": False,
    "filters": {
        "request_id": {
            "()": "question_app.logging.RequestIdFilter",
            "max_len": 8,
        },
    },
    "formatters": {
        "default": {
            "()": "uvicorn.logging.DefaultFormatter",
            "datefmt": "%H:%M:%S",
            "fmt": "%(levelprefix)s %(asctime)s %(message)s",
            "use_colors": None,
        },
        "access": {
            "()": "uvicorn.logging.AccessFormatter",
            "datefmt": "%H:%M:%S",
            "fmt": '%(levelprefix)s %(asctime)s %(request_id)s %(client_addr)s - "%(request_line)s" %(status_code)s',
        },
        "debug": {
            "()": "uvicorn.logging.DefaultFormatter",
            "datefmt": "%H:%M:%S",
            "fmt": "%(levelprefix)s %(asctime)s %(request_id)s %(name)s:%(lineno)d %(message)s",
        },
    },
    "handlers": {
        "default": {
            "filters": ["request_id"],
            "formatter": "default",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stderr",
        },
        "access": {
            "filters": ["request_id"],
            "formatter": "access",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
        },
        "debug": {
            "filters": ["request_id"],
            "formatter": "debug",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stderr",
        },
    },
    "loggers": {
        "uvicorn": {"handlers": ["default"], "level": "INFO", "propagate": False},
        "uvicorn.error": {"level": "INFO"},
        "uvicorn.access": {"handlers": ["access"], "level": "INFO", "propagate": False},
        "question_app": {"handlers": ["debug"], "level": "DEBUG"},
    },
}
