"""Shared CPU inference for the RT-DETRv2 ONNX detectors (fast_layout / fast_table).

These are lightweight ~20M-param object detectors exported to ONNX (see datalab training
repo: training/models/rtdetr/export_onnx.py). Inference is pure onnxruntime + numpy — no
torch / transformers dependency on the hot path. Preprocessing and post-processing exactly
mirror transformers' RTDetrImageProcessorFast so CPU outputs match the trained model.
"""

from __future__ import annotations

import json
import os
from typing import List

import numpy as np
from PIL import Image

_PROVIDERS = ["CPUExecutionProvider"]


class RTDetrOnnx:
    """Loads an exported RT-DETRv2 detector (model.onnx + config.json) and runs CPU detection.

    config.json carries id2label, the square input size, and the preprocessing params captured
    off the trained processor (rescale factor + optional ImageNet normalize)."""

    def __init__(self, model_dir: str, num_threads: int | None = None):
        import onnxruntime as ort

        with open(os.path.join(model_dir, "config.json")) as f:
            cfg = json.load(f)
        self.id2label = {int(k): v for k, v in cfg["id2label"].items()}
        pre = cfg["preprocess"]
        self.size = int(pre["image_size"])
        self.rescale = float(pre.get("rescale_factor", 1 / 255))
        self.do_normalize = bool(pre.get("do_normalize", False))
        self.mean = np.array(
            pre.get("image_mean", [0, 0, 0]), dtype=np.float32
        ).reshape(3, 1, 1)
        self.std = np.array(pre.get("image_std", [1, 1, 1]), dtype=np.float32).reshape(
            3, 1, 1
        )

        so = ort.SessionOptions()
        if num_threads:
            so.intra_op_num_threads = int(num_threads)
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            os.path.join(model_dir, "model.onnx"), sess_options=so, providers=_PROVIDERS
        )
        self._in = self.session.get_inputs()[0].name

    def _preprocess(self, image: Image.Image) -> np.ndarray:
        # RT-DETR stretches to a fixed square (bilinear), rescales to [0,1], optionally normalizes.
        img = image.convert("RGB").resize(
            (self.size, self.size), Image.Resampling.BILINEAR
        )
        arr = np.asarray(img, dtype=np.float32).transpose(2, 0, 1) * self.rescale
        if self.do_normalize:
            arr = (arr - self.mean) / self.std
        return arr

    def detect(
        self, images: List[Image.Image], threshold: float = 0.4, batch_size: int = 8
    ) -> List[List[dict]]:
        """Returns, per image, a list of {label, label_id, score, bbox:[x0,y0,x1,y1] pixels}."""
        out: List[List[dict]] = []
        for s in range(0, len(images), batch_size):
            chunk = images[s : s + batch_size]
            batch = np.stack([self._preprocess(im) for im in chunk], axis=0)
            logits, pred_boxes = self.session.run(None, {self._in: batch})
            for i, im in enumerate(chunk):
                out.append(
                    self._postprocess(
                        logits[i], pred_boxes[i], im.width, im.height, threshold
                    )
                )
        return out

    def _postprocess(
        self,
        logits: np.ndarray,
        boxes_cxcywh: np.ndarray,
        w: int,
        h: int,
        threshold: float,
    ) -> List[dict]:
        # mirrors RTDetrImageProcessorFast.post_process_object_detection (use_focal_loss=True):
        # cxcywh->xyxy, scale to (w,h), sigmoid, top-`num_queries` over (query x class), filter.
        cx, cy, bw, bh = (
            boxes_cxcywh[:, 0],
            boxes_cxcywh[:, 1],
            boxes_cxcywh[:, 2],
            boxes_cxcywh[:, 3],
        )
        xyxy = np.stack(
            [
                (cx - bw / 2) * w,
                (cy - bh / 2) * h,
                (cx + bw / 2) * w,
                (cy + bh / 2) * h,
            ],
            axis=1,
        )
        num_queries, num_classes = logits.shape
        scores = 1.0 / (1.0 + np.exp(-logits))  # sigmoid
        flat = scores.reshape(-1)
        k = num_queries
        top = np.argpartition(flat, -k)[-k:]  # top-k over query*class
        labels = top % num_classes
        qidx = top // num_classes
        sc = flat[top]
        keep = sc > threshold
        res = []
        for q, lbl, score in zip(qidx[keep], labels[keep], sc[keep]):
            b = xyxy[q]
            res.append(
                {
                    "label": self.id2label[int(lbl)],
                    "label_id": int(lbl),
                    "score": float(score),
                    "bbox": [float(b[0]), float(b[1]), float(b[2]), float(b[3])],
                }
            )
        return res


def resolve_model_dir(checkpoint: str) -> str:
    """Resolve a fast-model checkpoint to a local dir. Supports a plain local path, an
    ``hf://<repo>/<subfolder>`` ref (downloaded from the Hub), or an ``s3://`` path."""
    if checkpoint and checkpoint.startswith("hf://"):
        from huggingface_hub import snapshot_download

        parts = checkpoint[len("hf://") :].split("/")
        repo_id = "/".join(parts[:2])
        subfolder = "/".join(parts[2:])
        local = snapshot_download(
            repo_id,
            allow_patterns=[f"{subfolder}/*"] if subfolder else None,
        )
        return os.path.join(local, subfolder) if subfolder else local
    if checkpoint and os.path.isdir(checkpoint):
        return checkpoint
    if checkpoint and checkpoint.startswith("s3://"):
        from surya.common.s3 import download_directory  # type: ignore

        return download_directory(checkpoint)
    raise FileNotFoundError(
        f"fast-model checkpoint not found as a local dir: {checkpoint!r}"
    )
