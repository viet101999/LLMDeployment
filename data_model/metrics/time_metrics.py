from abc import ABC

import prometheus_client as prom

from data_model.metrics.base_metrics import BaseMetrics


class TimeMetrics(BaseMetrics, ABC):
    def __init__(self, registry, **kwargs) -> None:
        super(TimeMetrics, self).__init__(registry=registry)
        self.time_metrics_gauge = prom.Gauge(
            name='time_metrics_gauge',
            documentation='Inferences time metrics',
            labelnames=['time_type'],
            registry=self.registry
        )