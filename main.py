#main.py
from __future__ import annotations

import logging

from fastapi import (
    FastAPI,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.responses import RedirectResponse

from dependencies import ApiKeyDependency, get_config
from logger import setup_logger
from routers.chat import (
    router as chat_router,
)
from routers.misc import (
    router as misc_router,
)
from routers.models import (
    router as models_router,
)
from routers.realtime.rtc import (
    router as realtime_rtc_router,
)
from routers.realtime.ws import (
    router as realtime_ws_router,
)
from routers.speech import (
    router as speech_router,
)
from routers.stt import (
    router as stt_router,
)
from routers.vad import (
    router as vad_router,
)
from routers.custom_voice import (
    router as custom_voices_router,
)
# https://swagger.io/docs/specification/v3_0/grouping-operations-with-tags/
# https://fastapi.tiangolo.com/tutorial/metadata/#metadata-for-tags
TAGS_METADATA = [
    {"name": "automatic-speech-recognition"},
    {"name": "speech-to-text"},
    {"name": "realtime"},
    {"name": "models"},
    {"name": "diagnostic"},
    {
        "name": "experimental",
        "description": "Not meant for public use yet. May change or be removed at any time.",
    },
]


def create_app() -> FastAPI:
    config = get_config()  # HACK
    setup_logger(config.log_level)
    logger = logging.getLogger(__name__)

    logger.debug(f"Config: {config}")

    dependencies = []
    if config.api_key is not None:
        dependencies.append(ApiKeyDependency)

    app = FastAPI(dependencies=dependencies, openapi_tags=TAGS_METADATA)

    app.include_router(chat_router)
    app.include_router(stt_router)
    app.include_router(models_router)
    app.include_router(misc_router)
    app.include_router(realtime_rtc_router)
    app.include_router(realtime_ws_router)
    app.include_router(speech_router)
    app.include_router(vad_router)
    app.include_router(custom_voices_router)
    # HACK: move this elsewhere
    app.get("/v1/realtime", include_in_schema=False)(lambda: RedirectResponse(url="/v1/realtime/"))
    app.mount("/v1/realtime", StaticFiles(directory="realtime-console/dist", html=True))

    if config.allow_origins is not None:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=config.allow_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )


    return app
