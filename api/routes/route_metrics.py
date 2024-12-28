from fastapi import APIRouter, Request
from starlette.responses import Response
from prometheus_client import generate_latest, Counter, Summary, Gauge
import psutil
import torch
import time

router = APIRouter()

# Define Prometheus metrics
requests_counter = Counter('http_requests_total', 'Total HTTP requests received')
request_duration = Summary('request_duration_seconds', 'Request duration in seconds')
vram_usage_gauge = Gauge('vram_usage_gb', 'VRAM usage in GB')
cpu_ram_usage_gauge = Gauge('cpu_ram_usage_gb', 'CPU RAM usage in GB')

def get_vram_usage():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        return torch.cuda.memory_allocated() / (1024 ** 3)  # Convert to GB
    return None

def get_cpu_ram_usage():
    process = psutil.Process()
    return process.memory_info().rss / (1024 ** 3)  # Convert to GB

@router.middleware("http")
async def count_requests(request: Request, call_next):
    requests_counter.inc()
    with request_duration.time():
        response = await call_next(request)
    return response

@router.get("/metrics")
async def metrics():
    vram_usage = get_vram_usage()
    cpu_ram_usage = get_cpu_ram_usage()

    if vram_usage is not None:
        vram_usage_gauge.set(vram_usage)
    cpu_ram_usage_gauge.set(cpu_ram_usage)

    return Response(generate_latest(), media_type="text/plain")
