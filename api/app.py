import logging

import uvicorn
from dependency_injector.wiring import Provide, inject
from fastapi import FastAPI

from api.routes import health, route, route_metrics
from common.common_keys import *
from containers.app_container import AppContainer
from controller.llm_controller import LLMController


@inject
def create_app(
        app: FastAPI = Provide[AppContainer.app],
) -> FastAPI:
    app.include_router(route.router)
    app.include_router(health.router)
    app.include_router(route_metrics.router)

    @app.on_event("startup")
    async def startup_event():
        start_woker()
        logging.getLogger(FastAPI.__name__).info("App Start Successfully")

    return app

@inject
def start_woker(
        llm_controller: LLMController = Provide[AppContainer.llm_controller]
):
    logging.getLogger(FastAPI.__name__).info("Start woker")

@inject
def start_app(
        server_config: dict = Provide[AppContainer.server_config]
):
    app = create_app()
    uvicorn.run(
        app,
        host=server_config[HOST],
        port=int(server_config[PORT]),
        reload=False,
        log_level="debug",
        workers=1,
        factory=False,
        loop="asyncio",
        timeout_keep_alive=120
    )
