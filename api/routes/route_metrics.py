import psutil
import asyncio
import torch
import prometheus_client as prom
from typing import Dict

from concurrent.futures import ThreadPoolExecutor
from dependency_injector.wiring import Provide, inject
from fastapi import APIRouter, Depends, Response

from containers.registry_container import RegistryContainer
from data_model.metrics.time_metrics import TimeMetrics

EXECUTOR = ThreadPoolExecutor(max_workers=1)


async def get_total_cpu_memory_by_process() -> Dict[str, float]:
    """
    Get CPU, RAM by processes
    :return:
    """

    def _get_cpu_mem() -> Dict[str, float]:
        output = {}

        for proc in psutil.process_iter():
            mem = proc.memory_info().rss / 1024 ** 3    # Total RAM (in GB) used by all running processes
            cpu = proc.cpu_percent()                    # Total CPU usage (sum of percentages) across all processes
            output["RAM"] = output.get("RAM", 0) + mem
            output["CPU"] = output.get("CPU", 0) + cpu
        return output

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor=EXECUTOR, func=_get_cpu_mem)

async def get_vram_usage() -> Dict[str, float]:
    """
    Get VRAM usage for the current GPU
    :return: Dictionary with VRAM usage
    """

    def _get_vram() -> Dict[str, float]:
        output = {"VRAM": 0.0}
        if torch.cuda.is_available():
            torch.cuda.synchronize()  # Ensure all operations are completed
            vram = torch.cuda.memory_allocated() / (1024 ** 3)  # Convert to GB
            output["VRAM"] = vram
        return output

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor=EXECUTOR, func=_get_vram)

async def get_system_usage_metrics() -> Dict[str, float]:
    """
    Get system usage metrics for CPU, RAM, and VRAM
    :return: Dictionary with CPU, RAM, and VRAM usage
    """
    cpu_mem = await get_total_cpu_memory_by_process()
    vram = await get_vram_usage()

    # Merge CPU, RAM, and VRAM metrics
    return {**cpu_mem, **vram}

router = APIRouter(
    tags=["CallCenter"],
    responses={404: {"description": "Not found"}, 500: {"description": "server error"}}
)


@router.get(
    path="/metrics",
    tags=["CallCenter"]
)
@inject
async def metrics(
        time_metrics: TimeMetrics = Depends(Provide[RegistryContainer.TimeMetrics])
) -> Response:
    """
    get metrics to visualize
    :param time_metrics: time metrics
    :return:
    """
    registry = time_metrics.registry

    for label, scrapper_function in time_metrics.metrics.items():
        time_metrics.system_usage.labels(label).set(scrapper_function())

    process_info = await get_system_usage_metrics()

    for metric, value in process_info.items():
        time_metrics.system_usage.labels(metric).set(value)

    return Response(
        content=prom.generate_latest(registry),
        media_type="text/plain"
    )