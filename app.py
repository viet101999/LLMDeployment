from api.app import start_app
from containers.app_container import AppContainer
from setting import setup_server

app_container = AppContainer()

if __name__ == "__main__":
    setup_server()
    start_app()
