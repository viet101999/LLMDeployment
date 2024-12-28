from abc import ABC

import prometheus_client as prom

from data_model.metrics.base_metrics import BaseMetrics


class TimeMetrics(BaseMetrics, ABC):
    def __init__(self, registry, **kwargs) -> None:
        super(TimeMetrics, self).__init__(registry=registry)
        # self.processing_time_summary = prom.Histogram(
        #     name='processing_time_seconds',
        #     documentation='Time spent processing files',
        #     buckets=list(range(0, 3600, 30)),
        #     labelnames=['time_type'],
        #     registry=self.registry
        # )

        # self.delay_time_histogram = prom.Histogram(
        #     name='delay_time_seconds',
        #     documentation='Delay time in seconds between download and processing',
        #     buckets=list(range(0, 3600, 30)),
        #     registry=self.registry
        # )
        
        self.time_metrics_gauge = prom.Gauge(
            name='time_metrics_gauge',
            documentation='Waiting and processing time of a file',
            labelnames=['time_type'],
            registry=self.registry
        )
        
        self.time_metrics_histogram = prom.Histogram(
            name='time_metrics_histogram',
            documentation='Waiting and processing time of a file',
            labelnames=['time_type'],
            buckets=list(range(0, 3600, 30)),
            registry=self.registry
        )