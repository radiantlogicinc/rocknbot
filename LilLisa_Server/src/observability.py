import logging

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

from opentelemetry._logs import set_logger_provider
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.exporter.otlp.proto.grpc._log_exporter import OTLPLogExporter

from opentelemetry.metrics import set_meter_provider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter

# Initialize OpenTelemetry
# Traces
trace.set_tracer_provider(TracerProvider())
otlp_exporter = OTLPSpanExporter()
span_processor = BatchSpanProcessor(otlp_exporter)
trace.get_tracer_provider().add_span_processor(span_processor)

#Logs
logger_provider = LoggerProvider()
set_logger_provider(logger_provider)
logger_provider.add_log_record_processor(BatchLogRecordProcessor(OTLPLogExporter()))
handler = LoggingHandler(level=logging.NOTSET, logger_provider=logger_provider)
# Attach OTLP handler to root logger
logging.getLogger().addHandler(handler)

# Metrics
metrics_exporter = OTLPMetricExporter()
reader = PeriodicExportingMetricReader(metrics_exporter)
meter_provider = MeterProvider(metric_readers=[reader])
set_meter_provider(meter_provider)

# Custom metrics
meter = meter_provider.get_meter(__name__)
metrics_fastapi_requests_total = meter.create_counter("fastapi_requests_total")
metrics_fastapi_responses_total = meter.create_counter("fastapi_responses_total")
metrics_fastapi_requests_duration_seconds = meter.create_histogram("fastapi_requests_duration_seconds")
metrics_fastapi_requests_in_progress = meter.create_up_down_counter("fastapi_requests_in_progress")