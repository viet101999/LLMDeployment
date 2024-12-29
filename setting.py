from dependency_injector.wiring import Provide, inject

from containers.app_container import AppContainer
from utils.logging_utils import setup_logging


@inject
def setup_server(
        logging_config=Provide[AppContainer.logging_config]
):
    setup_logging(
        logging_folder=logging_config["folder"],
        log_name=logging_config["name"]
    )
