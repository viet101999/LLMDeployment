from dependency_injector import containers, providers

from data_model.metrics.base_metrics import BaseMetrics
from data_model.metrics.time_metrics import TimeMetrics
from prometheus_client import CollectorRegistry


class RegistryContainer(containers.DeclarativeContainer):
    wiring_config = containers.WiringConfiguration(
        modules=[
            "api.routes.route_metrics",
        ]
    )
    registry = providers.Singleton(CollectorRegistry)
    TimeMetrics = providers.Singleton(TimeMetrics,
                                       registry=registry)

    @classmethod
    def metrics_factory(cls, metrics_type: str) -> BaseMetrics:
        """
        Instantiate different types of metrics
        :param metrics_type: metric type
        :return:
        """
        if hasattr(cls, metrics_type):
            metrics_provider = getattr(cls, metrics_type)
            metrics = metrics_provider()
            return metrics
        else:
            raise ValueError(f"{metrics_type} not exist")
