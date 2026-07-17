"""FastLayoutPredictor — rf-detr page-layout detector, served from one shared instance.

Drop-in alternative to surya.layout.LayoutPredictor: same LayoutResult/LayoutBox output, but
a lightweight rf-detr object detector instead of the VLM. Labels are canonicalized through the
same LAYOUT_PRED_RELABEL map the VLM layout model uses, so downstream consumers (marker) are unchanged.

The model always runs in a single shared server process (see surya.fast_layout.server);
FastLayoutPredictor is a thin client of it. This is the only path: N worker processes
(e.g. marker) would otherwise each load their own model and thread pool and thrash the
CPU/GPU — there's no benefit to more than one layout model on a host. The first client to
run attaches to a running server or spawns one; the rest attach.
"""

from __future__ import annotations

from typing import List, Optional

from PIL import Image

from surya.layout.label import LAYOUT_PRED_RELABEL
from surya.layout.schema import LayoutBox, LayoutResult
from surya.logging import get_logger
from surya.settings import settings

logger = get_logger()


def _poly(b):
    x0, y0, x1, y1 = b
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def build_layout_result(image: Image.Image, dets, order) -> LayoutResult:
    """Turn one page's raw rf-detr detections into an ordered LayoutResult.

    `dets` is the per-image detection list from ``RfDetrTorch.detect`` (optionally
    carrying an encoder feature map on ``.features``). `order` is an OrderPredictor
    or None. Shared by the in-process predictor and the server so ordering/relabel
    behaviour is identical on both paths.
    """
    # Reading order: the learned AR head (cross-attends to the encoder feature map) when
    # available, else a top-to-bottom / left-to-right raster sort.
    feats = getattr(dets, "features", None)
    if order is not None and feats is not None and dets:
        positions = order.order_page(
            feats,
            [d["bbox"] for d in dets],
            [d["label"] for d in dets],
            image.width,
            image.height,
        )
    else:
        # Raster sort: the normal path when order is off for this call.
        if order is not None and feats is None and dets:
            # Order model loaded but no feature map came back — it should have run
            # but didn't. Surface this; the "model never loaded" case is logged
            # once at first load.
            logger.warning(
                "Reading-order model loaded but detector returned no feature map; "
                "falling back to raster sort for this page."
            )
        raster = sorted(
            range(len(dets)),
            key=lambda i: (dets[i]["bbox"][1], dets[i]["bbox"][0]),
        )
        positions = [0] * len(dets)
        for rank, i in enumerate(raster):
            positions[i] = rank
    boxes = []
    for d, pos in zip(dets, positions):
        raw = d["label"]
        boxes.append(
            LayoutBox(
                polygon=_poly(d["bbox"]),
                label=LAYOUT_PRED_RELABEL.get(raw, raw),
                raw_label=raw,
                position=pos,
                confidence=d["score"],
            )
        )
    boxes.sort(key=lambda b: b.position)
    return LayoutResult(
        bboxes=boxes,
        image_bbox=[0.0, 0.0, float(image.width), float(image.height)],
    )


class FastLayoutPredictor:
    """Thin client of the shared fast-layout server (surya.fast_layout.server).

    Holds no model — every call hands the batch to the one shared server, which
    owns the single rf-detr instance and does its own continuous batching across
    all clients. ``num_threads`` and ``batch_size`` are governed server-side
    (FAST_LAYOUT_NUM_THREADS / FAST_LAYOUT_SERVER_MAX_BATCH); the constructor
    still accepts ``num_threads`` for signature compatibility but it has no local
    effect here.
    """

    def __init__(
        self,
        checkpoint: Optional[str] = None,
        num_threads: Optional[int] = None,
        use_order: Optional[bool] = None,
    ):
        from surya.fast_layout.client import FastLayoutServerClient

        self.use_order = (
            settings.FAST_LAYOUT_USE_ORDER if use_order is None else use_order
        )
        self._disable_tqdm = settings.DISABLE_TQDM
        self._client = FastLayoutServerClient(checkpoint=checkpoint)

    def to(
        self, *args, **kwargs
    ):  # API parity with other predictors (no-op; device is set server-side)
        return

    def __call__(
        self,
        images: List[Image.Image],
        threshold: Optional[float] = None,
        batch_size: Optional[int] = None,  # ignored: server controls batching
        use_order: Optional[bool] = None,
    ) -> List[LayoutResult]:
        if not images:
            return []
        threshold = (
            settings.FAST_LAYOUT_CONFIDENCE_THRESHOLD
            if threshold is None
            else threshold
        )
        use_order = self.use_order if use_order is None else use_order
        return self._client(images, threshold=threshold, use_order=use_order)
