import os

from dependency_injector import containers, providers
from fastapi import FastAPI

from controller.llm_controller import LLMController
from containers.registry_container import RegistryContainer
from modules.model import ModelLoad

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class AppContainer(containers.DeclarativeContainer):
    wiring_config = containers.WiringConfiguration(
        modules=[
            "config",
            "setting",
            "api.app",
            "api.routes.route",
            "api.routes.route_metrics",
            "containers.registry_container",
        ]
    )
    config = providers.Configuration(json_files=["config/config.json"])
    logging_config = providers.Resource(config.logging)
    server_config = providers.Resource(config.server)
    model_config = providers.Resource(config.model_config)

    registry_container = providers.Container(
        RegistryContainer
    )
    app = providers.Singleton(FastAPI)
    llm_model = providers.Singleton(
        ModelLoad,
        config_app=model_config
    )
    llm_controller = providers.Singleton(
        LLMController,
        llm_model=llm_model
    )
