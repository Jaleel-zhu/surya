"""RfDetrTorch — rf-detr (Roboflow) detector via the vendored model copy (no rfdetr package).

Exposes the same ``.detect()`` interface as :class:`surya.common.rtdetr_onnx.RTDetrOnnx`
so ``fast_table`` / ``fast_layout`` are engine-agnostic. Inference goes through the slimmed,
detection-only model definition vendored under ``surya.common.rfdetr`` (validated byte-for-byte
against the upstream rfdetr package). Pure PyTorch — runs on cpu/mps/cuda.

Model dir layout (downloaded from datalab-to/surya_models):
  rfdetr_<task>.pth   the fine-tuned rf-detr weights
  config.json         {"arch": "rf-detr-large", "categories": [{"id", "name"}, ...], ...}
"""

from __future__ import annotations

import glob
import json
import os
from typing import List, Optional

from PIL import Image


class _DetList(list):
    """A list of detections that can also carry the page's encoder feature map (.features)."""

    features = None


def _pick_device(device: Optional[str]) -> str:
    import torch

    if device:
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class RfDetrTorch:
    def __init__(
        self,
        model_dir: str,
        num_threads: Optional[int] = None,
        device: Optional[str] = None,
    ):
        import torch

        if num_threads:
            torch.set_num_threads(int(num_threads))

        self.device = _pick_device(device)
        if self.device == "mps":
            # DINOv2 backbone hits a few ops without MPS kernels; fall back to CPU per-op
            # instead of crashing.
            os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

        with open(os.path.join(model_dir, "config.json")) as f:
            cfg = json.load(f)

        # rf-detr's predict() returns 0-indexed class ids that line up with the COCO
        # categories sorted by id (Row=0, Col=1; layout: Caption=0 ... Text=15).
        cats = sorted(cfg["categories"], key=lambda c: c["id"])
        self.id2label = {i: c["name"] for i, c in enumerate(cats)}

        weights = cfg.get("weights")
        weights = os.path.join(model_dir, weights) if weights else None
        if not weights or not os.path.exists(weights):
            pths = sorted(glob.glob(os.path.join(model_dir, "*.pth")))
            if not pths:
                raise FileNotFoundError(f"no rf-detr .pth weights found in {model_dir}")
            weights = pths[0]

        arch = (cfg.get("arch") or "rf-detr-large").lower()
        if "base" in arch:
            raise ValueError(
                "vendored rf-detr copy is rf-detr-large only; got arch=%r" % arch
            )
        from surya.common.rfdetr import RFDetrDetector

        # Honor resolution / PE overrides from config.json so reduced-resolution
        # fine-tunes (e.g. the 448 layout model) run at their trained size; absent
        # these keys the predictor falls back to LARGE_ARGS (704), so older configs
        # are unaffected.
        arch_args = {}
        if cfg.get("resolution"):
            arch_args["resolution"] = int(cfg["resolution"])
        if cfg.get("positional_encoding_size"):
            arch_args["positional_encoding_size"] = int(cfg["positional_encoding_size"])
        self.model = RFDetrDetector(
            weights_path=weights, device=self.device, arch_args=arch_args or None
        )

    def detect(
        self,
        images: List[Image.Image],
        threshold: float = 0.4,
        batch_size: int = 8,
        return_features: bool = False,
    ) -> List[List[dict]]:
        """Returns, per image, a list of {label, label_id, score, bbox:[x0,y0,x1,y1] pixels}.
        When return_features=True, each per-image list carries the encoder feature map on a
        ``.features`` attribute ([C,F,F] tensor) for the reading-order head."""
        out: List = []
        for s in range(0, len(images), batch_size):
            chunk = [im.convert("RGB") for im in images[s : s + batch_size]]
            for det in self.model.predict(
                chunk, threshold=threshold, return_features=return_features
            ):
                boxes, scores, labels = det["boxes"], det["scores"], det["labels"]
                dets: List[dict] = _DetList()
                for i in range(len(scores)):
                    cid = int(labels[i])
                    x0, y0, x1, y1 = (float(v) for v in boxes[i].tolist())
                    dets.append(
                        {
                            "label": self.id2label.get(cid, str(cid)),
                            "label_id": cid,
                            "score": float(scores[i]),
                            "bbox": [x0, y0, x1, y1],
                        }
                    )
                if return_features:
                    dets.features = det.get("features")
                out.append(dets)
        return out


def _find_onnx_dir(model_dir: str) -> Optional[str]:
    """Return the directory containing an exported ``model.onnx`` (alongside its
    ``config.json``). Checkpoints ship the onnx in a dated subfolder
    (e.g. ``2025_06/model.onnx``) next to the legacy ``.pth`` weights, so look
    one level down and prefer the most recent if there are several."""
    if os.path.exists(os.path.join(model_dir, "model.onnx")):
        return model_dir
    candidates = sorted(
        os.path.dirname(p)
        for p in glob.glob(os.path.join(model_dir, "*", "model.onnx"))
    )
    return candidates[-1] if candidates else None


def load_detector(
    model_dir: str, num_threads: Optional[int] = None, device: Optional[str] = None
):
    """Pick the inference engine from what's in the model dir: rf-detr ``.pth`` weights
    (torch; cuda/mps/cpu) or an exported ``model.onnx`` (RT-DETRv2, pure onnxruntime, CPU).

    The torch ``.pth`` are the validated, in-use weights and are preferred. The ONNX
    export is used only as a fallback when no ``.pth`` is present (the current dated
    exports score ~0 on real pages — broken/stale — so they must not shadow the .pth).
    Set ``SURYA_FORCE_ONNX_DETECTOR=1`` to force the ONNX engine once the export is fixed.
    On CPU the torch rf-detr runs ~0.3s/page, so it remains the fast-mode path."""
    force_onnx = os.environ.get("SURYA_FORCE_ONNX_DETECTOR", "").lower() in (
        "1",
        "true",
        "yes",
    )
    has_pth = bool(glob.glob(os.path.join(model_dir, "*.pth")))
    if has_pth and not force_onnx:
        return RfDetrTorch(model_dir, num_threads=num_threads, device=device)
    onnx_dir = _find_onnx_dir(model_dir)
    if onnx_dir is not None:
        from surya.common.rtdetr_onnx import RTDetrOnnx

        return RTDetrOnnx(onnx_dir, num_threads=num_threads)
    return RfDetrTorch(model_dir, num_threads=num_threads, device=device)
