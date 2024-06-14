from locust import events, stats

# set customed percentiles for statistics and charts
stats.PERCENTILES_TO_STATISTICS = [0.5, 0.75, 0.9, 0.99]
stats.PERCENTILES_TO_CHART = [0.5, 0.75, 0.9, 0.99]


def update_custom_metric(name, value, length_value=0):
    events.request.fire(
        request_type="METRIC",
        name=name,
        response_time=value,
        response_length=length_value,
        exception=None,
        context=None,
    )
