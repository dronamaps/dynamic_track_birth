import os

from celery import Celery


REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")

celery_app = Celery(
    "crop_row_gap_analyzer",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=["backend.tasks"],
)
celery_app.conf.update(
    task_track_started=True,
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    result_expires=24 * 60 * 60,
    worker_prefetch_multiplier=1,
    task_acks_late=True,
)
