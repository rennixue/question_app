import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException

from .middlewares import RequestIdMiddleware
from .routers import router

logger = logging.getLogger(__name__)

app = FastAPI()
app.include_router(router)
app.add_middleware(RequestIdMiddleware)


@app.exception_handler(RequestValidationError)
async def handle_request_validation_error(request: Request, exc: RequestValidationError):
    logger.error("%r: %r", repr(exc), await request.body())
    return JSONResponse({"error": str(exc)}, 422)


@app.exception_handler(HTTPException)
async def handle_http_exception(request: Request, exc: HTTPException):
    return JSONResponse({"error": str(exc.detail)}, exc.status_code)


@app.exception_handler(Exception)
async def handle_exception(request: Request, exc: Exception):
    logger.error(repr(exc))
    return JSONResponse({"error": "internal server error"}, 500)


def run_dev():
    from pathlib import Path

    import dotenv
    import uvicorn

    from question_app.logging import DEV_LOG_CFG

    dotenv.load_dotenv()

    uvicorn.run("question_app.app:app", reload=True, reload_dirs=[str(Path(__file__).parent)], log_config=DEV_LOG_CFG)
