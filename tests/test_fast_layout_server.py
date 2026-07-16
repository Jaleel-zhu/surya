"""Fast-layout shared server: engine correctness + continuous batching.

Exercises the real rf-detr model, so it skips gracefully when it can't be fetched
(offline CI), matching how the VLM tests skip without a backend.
"""

import threading

import pytest

from surya.settings import settings


@pytest.fixture(scope="module")
def engine():
    server = pytest.importorskip("surya.fast_layout.server")
    try:
        return server.LayoutEngine()
    except Exception as exc:  # model download / load unavailable
        pytest.skip(f"fast-layout model unavailable: {exc}")


def test_engine_run_batch_orders_boxes(engine, test_image):
    """A page run through the engine comes back as a valid, ordered LayoutResult."""
    res = engine.run_batch(
        [test_image],
        [{"threshold": settings.FAST_LAYOUT_CONFIDENCE_THRESHOLD, "use_order": True}],
    )[0]
    assert res.image_bbox == [
        0.0,
        0.0,
        float(test_image.width),
        float(test_image.height),
    ]
    for b in res.bboxes:
        assert b.label
        assert isinstance(b.position, int)
    assert sorted(b.position for b in res.bboxes) == list(range(len(res.bboxes)))


def test_continuous_batching_coalesces(engine, test_image, monkeypatch):
    """Pages submitted concurrently (as N client requests would) merge into a
    single batched detect() call rather than running one at a time."""
    from surya.common.batch_service.server import _Batcher
    from surya.fast_layout.config import layout_service_config

    config = layout_service_config()
    monkeypatch.setattr(config, "batch_wait_ms", 50, raising=False)

    sizes = []
    orig = engine.model.detect

    def spy(images, **kw):
        sizes.append(len(images))
        return orig(images, **kw)

    monkeypatch.setattr(engine.model, "detect", spy)

    batcher = _Batcher(engine, config)

    def submit():
        job = batcher.submit(
            test_image,
            {"threshold": settings.FAST_LAYOUT_CONFIDENCE_THRESHOLD, "use_order": True},
        )
        assert job.done.wait(120)
        assert job.error is None

    threads = [threading.Thread(target=submit) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert sum(sizes) == 6
    assert max(sizes) > 1  # at least some pages coalesced into one forward
