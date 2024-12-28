from fastapi import FastAPI
from api.app import initialize_routes
from prometheus_client import start_http_server
import threading

# Initialize FastAPI
app = FastAPI()

# Initialize routes
initialize_routes(app)

# Start Prometheus metrics server
def start_metrics_server():
    start_http_server(8001)  # Expose metrics on a separate port
threading.Thread(target=start_metrics_server, daemon=True).start()

@app.on_event("startup")
async def on_startup():
    print("Application has started.")

@app.on_event("shutdown")
async def on_shutdown():
    print("Application has stopped.")
