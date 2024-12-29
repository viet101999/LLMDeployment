import psutil
import prometheus_client as prom

from abc import ABC

from prometheus_client import CollectorRegistry


class BaseMetrics(ABC):
    def __init__(self, registry, **kwargs) -> None:
        super(BaseMetrics, self).__init__()
        self.registry = registry
        self.system_usage = prom.Gauge(
            name='system_usage',
            documentation='Hold current system resource usage',
            labelnames=['resource_type'],
            registry=self.registry
        )
        self.metrics = {
            "GlobalCPU": psutil.cpu_percent,                # Total CPU usage percentage across the system
            "GlobalRAM": lambda: psutil.virtual_memory()[2] # Total RAM usage percentage across the system
        }