from fastapi import FastAPI
from api.routes.route import router as route_router
from api.routes.route_metrics import router as metrics_router
from api.routes.health import router as health_router

def initialize_routes(app: FastAPI):
    # Include all routers
    app.include_router(route_router)
    app.include_router(metrics_router)
    app.include_router(health_router)
