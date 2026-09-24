"""
RoadGuard AI — Driver Drowsiness Classifier & Temporal Buffer Suite
Provides production inference for trained PyTorch driver drowsiness model
with explicit class mapping metadata, input validation, and temporal smoothing buffer.
"""

from __future__ import annotations

import time
from collections import deque
from pathlib import Path
from typing import Dict, Any, Optional, Union, List

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms


class CustomDriverCNN(nn.Module):
    """
    Lightweight custom 4-stage convolutional neural network for driver drowsiness classification.
    Parameter count: ~250K. Optimized for ultra-low latency edge/mobile ADAS inference.
    """

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1 (224 -> 112)
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 2 (112 -> 56)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 3 (56 -> 28)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 4 (28 -> 14)
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.35),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        logits = self.classifier(feat)
        return logits.squeeze(-1)


class DriverClassifier:
    """
    Inference adapter for the PyTorch Driver Drowsiness Model.
    Loads trained weights, processes raw BGR/RGB image inputs, and computes
    explicitly mapped drowsiness probability with class metadata.
    """

    # Deployment-safe model path:
    # backend/core/driver_classifier.py
    #          ↓
    # backend/model_assets/best_driver_model.pt
    DEFAULT_WEIGHTS_PATH = (
        Path(__file__).resolve().parents[1]
        / "model_assets"
        / "best_driver_model.pt"
    )

    # Explicit class mapping ground truth
    CLASS_TO_IDX = {
        "non_drowsy": 0,
        "drowsy": 1
    }

    IDX_TO_CLASS = {
        0: "NON_DROWSY",
        1: "DROWSY"
    }

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
        class_to_idx: Optional[Dict[str, int]] = None,
    ):
        self.model_path = (
            Path(model_path)
            if model_path
            else self.DEFAULT_WEIGHTS_PATH
        )

        self.class_to_idx = class_to_idx or self.CLASS_TO_IDX
        self.idx_to_class = {
            v: k.upper()
            for k, v in self.class_to_idx.items()
        }

        # Select device: MPS -> CUDA -> CPU
        if device:
            self.device = torch.device(device)

        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")

        elif torch.cuda.is_available():
            self.device = torch.device("cuda")

        else:
            self.device = torch.device("cpu")

        # Standard ImageNet normalization pipeline for 224x224 input
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        self.model = CustomDriverCNN().to(self.device)

        self._load_weights()

        self.model.eval()

        print(
            f"[DriverClassifier] ✅ Model loaded from "
            f"{self.model_path} on {self.device}"
        )

    def _load_weights(self) -> None:
        """Loads state dict from disk."""

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Drowsiness model weights not found at: "
                f"{self.model_path}"
            )

        checkpoint = torch.load(
            self.model_path,
            map_location=self.device
        )

        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:

            self.model.load_state_dict(
                checkpoint["state_dict"]
            )

            if "class_to_idx" in checkpoint:
                self.class_to_idx = checkpoint["class_to_idx"]

                self.idx_to_class = {
                    v: k.upper()
                    for k, v in self.class_to_idx.items()
                }

        elif (
            isinstance(checkpoint, dict)
            and "model_state_dict" in checkpoint
        ):

            self.model.load_state_dict(
                checkpoint["model_state_dict"]
            )

            if "class_to_idx" in checkpoint:
                self.class_to_idx = checkpoint["class_to_idx"]

                self.idx_to_class = {
                    v: k.upper()
                    for k, v in self.class_to_idx.items()
                }

        elif isinstance(checkpoint, dict):

            self.model.load_state_dict(checkpoint)

        else:

            self.model = checkpoint

    def _preprocess(
        self,
        image: Union[np.ndarray, Image.Image]
    ) -> torch.Tensor:
        """
        Converts BGR numpy image or PIL Image to normalized
        PyTorch tensor (1, 3, 224, 224).
        """

        if isinstance(image, np.ndarray):

            if (
                len(image.shape) == 3
                and image.shape[2] == 3
            ):
                rgb_img = cv2.cvtColor(
                    image,
                    cv2.COLOR_BGR2RGB
                )

            else:
                rgb_img = image

            pil_img = Image.fromarray(rgb_img)

        else:
            pil_img = image

        return (
            self.transform(pil_img)
            .unsqueeze(0)
            .to(self.device)
        )

    def predict_proba(
        self,
        image: Union[np.ndarray, Image.Image]
    ) -> float:
        """
        Computes raw probability P(Drowsy) for a single
        input frame / face crop.

        Returns:
            float probability between
            0.0 (Alert / Non-Drowsy)
            and 1.0 (Drowsy).
        """

        tensor = self._preprocess(image)

        with torch.no_grad():

            logit = self.model(tensor)

            # Binary sigmoid probability for class 1
            prob_raw = torch.sigmoid(logit).item()

        drowsy_idx = self.class_to_idx.get(
            "drowsy",
            1
        )

        if drowsy_idx == 1:
            p_drowsy = prob_raw

        else:
            p_drowsy = 1.0 - prob_raw

        return float(
            max(
                0.0,
                min(1.0, p_drowsy)
            )
        )

    def predict(
        self,
        image: Union[np.ndarray, Image.Image],
        threshold: float = 0.50,
        return_debug: bool = False
    ) -> Dict[str, Any]:
        """
        Runs single-frame inference and returns structured
        prediction results with latency and explicit
        class semantics.
        """

        t0 = time.perf_counter()

        tensor = self._preprocess(image)

        with torch.no_grad():

            logit_val = self.model(
                tensor
            ).item()

            prob_raw = torch.sigmoid(
                torch.tensor(logit_val)
            ).item()

        drowsy_idx = self.class_to_idx.get(
            "drowsy",
            1
        )

        if drowsy_idx == 1:

            p_drowsy = prob_raw
            p_non_drowsy = 1.0 - prob_raw

        else:

            p_drowsy = 1.0 - prob_raw
            p_non_drowsy = prob_raw

        latency_ms = (
            time.perf_counter() - t0
        ) * 1000.0

        is_drowsy = (
            p_drowsy >= threshold
        )

        pred_class = (
            "DROWSY"
            if is_drowsy
            else "NON_DROWSY"
        )

        pred_prob = (
            p_drowsy
            if is_drowsy
            else p_non_drowsy
        )

        out = {
            "cnn_drowsy_probability":
                round(p_drowsy, 4),

            "cnn_predicted_class":
                pred_class,

            "cnn_predicted_class_probability":
                round(pred_prob, 4),

            # Backward-compatible fields
            "drowsy_probability":
                round(p_drowsy, 4),

            "is_drowsy":
                is_drowsy,

            "label":
                pred_class,

            "confidence":
                round(pred_prob, 4),

            "latency_ms":
                round(latency_ms, 2)
        }

        if return_debug:

            out["cnn_debug"] = {

                "class_to_idx":
                    self.class_to_idx,

                "raw_logit":
                    round(logit_val, 4),

                "probabilities": {

                    "drowsy":
                        round(p_drowsy, 4),

                    "non_drowsy":
                        round(p_non_drowsy, 4),
                },

                "tensor_shape":
                    list(tensor.shape),

                "tensor_min":
                    round(float(tensor.min()), 4),

                "tensor_max":
                    round(float(tensor.max()), 4),

                "tensor_mean":
                    round(float(tensor.mean()), 4),
            }

        return out


class TemporalDriverBuffer:
    """
    Temporal aggregation and hysteresis buffer for driver
    drowsiness states.

    Prevents instantaneous false-positive alert fluttering
    by tracking a rolling window and enforcing consecutive
    frame thresholding.
    """

    def __init__(
        self,
        window_size: int = 15,
        drowsy_threshold: float = 0.55,
        consecutive_frames_required: int = 5
    ):
        self.window_size = window_size
        self.drowsy_threshold = drowsy_threshold
        self.consecutive_frames_required = (
            consecutive_frames_required
        )

        self.history: deque = deque(
            maxlen=window_size
        )

        self.consecutive_drowsy_count: int = 0

        self.current_state: str = "ALERT"

    def push(
        self,
        prob: float
    ) -> Dict[str, Any]:
        """
        Pushes a new frame drowsiness probability into
        the buffer and returns smoothed temporal stats.
        """

        self.history.append(prob)

        if prob >= self.drowsy_threshold:

            self.consecutive_drowsy_count += 1

        else:

            self.consecutive_drowsy_count = max(
                0,
                self.consecutive_drowsy_count - 1
            )

        rolling_mean = float(
            np.mean(self.history)
        )

        rolling_max = float(
            np.max(self.history)
        )

        # Hysteresis state machine
        if (
            self.consecutive_drowsy_count
            >= self.consecutive_frames_required
            or rolling_mean >= 0.65
        ):

            self.current_state = "DROWSY"

        elif (
            rolling_mean >= 0.40
            or self.consecutive_drowsy_count >= 3
        ):

            self.current_state = "LOW_VIGILANCE"

        else:

            self.current_state = "ALERT"

        return {

            "raw_probability":
                round(prob, 4),

            "smoothed_fatigue":
                round(rolling_mean, 4),

            "peak_fatigue":
                round(rolling_max, 4),

            "consecutive_drowsy_frames":
                self.consecutive_drowsy_count,

            "state":
                self.current_state,

            "buffer_depth":
                len(self.history)
        }

    def reset(self) -> None:
        """Resets buffer state."""

        self.history.clear()

        self.consecutive_drowsy_count = 0

        self.current_state = "ALERT"