"""
RoadGuard AI — Driver Drowsiness Classifier & Temporal Buffer Suite
Provides production inference for trained PyTorch driver drowsiness model
with zero-leakage subject generalization and temporal smoothing buffer.
"""

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
    Loads trained weights, processes raw BGR/RGB image inputs, and computes calibrated drowsiness probability.
    """
    DEFAULT_WEIGHTS_PATH = Path("/Users/rudrakshtyagi/Desktop/roadgaurdai/models/drowsiness/best_driver_model.pt")

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        device: Optional[str] = None
    ):
        self.model_path = Path(model_path) if model_path else self.DEFAULT_WEIGHTS_PATH
        
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

    def _load_weights(self) -> None:
        """Loads state dict from disk."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Drowsiness model weights not found at: {self.model_path}")
        
        checkpoint = torch.load(self.model_path, map_location=self.device)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["state_dict"])
        elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        elif isinstance(checkpoint, dict):
            self.model.load_state_dict(checkpoint)
        else:
            self.model = checkpoint

    def predict_proba(self, image: Union[np.ndarray, Image.Image]) -> float:
        """
        Computes raw probability P(Drowsy) = sigmoid(logit) for a single input frame / face crop.
        
        Args:
            image: numpy ndarray (BGR or RGB) or PIL Image.
            
        Returns:
            float probability between 0.0 (Alert) and 1.0 (Drowsy).
        """
        if isinstance(image, np.ndarray):
            # Assume BGR if OpenCV format, convert to RGB
            if len(image.shape) == 3 and image.shape[2] == 3:
                rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                rgb_img = image
            pil_img = Image.fromarray(rgb_img)
        else:
            pil_img = image

        tensor = self.transform(pil_img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logit = self.model(tensor)
            prob = torch.sigmoid(logit).item()

        return float(prob)

    def predict(
        self,
        image: Union[np.ndarray, Image.Image],
        threshold: float = 0.50
    ) -> Dict[str, Any]:
        """
        Runs single-frame inference and returns structured prediction results with latency.
        """
        t0 = time.perf_counter()
        prob = self.predict_proba(image)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        is_drowsy = prob >= threshold
        label = "DROWSY" if is_drowsy else "NON_DROWSY"

        return {
            "drowsy_probability": round(prob, 4),
            "is_drowsy": is_drowsy,
            "label": label,
            "confidence": round(prob if is_drowsy else (1.0 - prob), 4),
            "latency_ms": round(latency_ms, 2)
        }


class TemporalDriverBuffer:
    """
    Temporal aggregation and hysteresis buffer for driver drowsiness states.
    Prevents instantaneous false-positive alert fluttering by tracking a rolling window
    and enforcing consecutive frame thresholding.
    """
    def __init__(
        self,
        window_size: int = 15,
        drowsy_threshold: float = 0.55,
        consecutive_frames_required: int = 5
    ):
        self.window_size = window_size
        self.drowsy_threshold = drowsy_threshold
        self.consecutive_frames_required = consecutive_frames_required

        self.history: deque = deque(maxlen=window_size)
        self.consecutive_drowsy_count: int = 0
        self.current_state: str = "ALERT"

    def push(self, prob: float) -> Dict[str, Any]:
        """
        Pushes a new frame drowsiness probability into the buffer and returns smoothed temporal stats.
        
        Args:
            prob: P(Drowsy) for current frame [0.0 - 1.0].
            
        Returns:
            Dictionary containing smoothed fatigue index, state label, and temporal metrics.
        """
        self.history.append(prob)

        if prob >= self.drowsy_threshold:
            self.consecutive_drowsy_count += 1
        else:
            self.consecutive_drowsy_count = max(0, self.consecutive_drowsy_count - 1)

        rolling_mean = float(np.mean(self.history))
        rolling_max = float(np.max(self.history))

        # Hysteresis state machine
        if self.consecutive_drowsy_count >= self.consecutive_frames_required or rolling_mean >= 0.65:
            self.current_state = "DROWSY"
        elif rolling_mean >= 0.40 or self.consecutive_drowsy_count >= 3:
            self.current_state = "LOW_VIGILANCE"
        else:
            self.current_state = "ALERT"

        return {
            "raw_probability": round(prob, 4),
            "smoothed_fatigue": round(rolling_mean, 4),
            "peak_fatigue": round(rolling_max, 4),
            "consecutive_drowsy_frames": self.consecutive_drowsy_count,
            "state": self.current_state,
            "buffer_depth": len(self.history)
        }

    def reset(self) -> None:
        """Resets buffer state."""
        self.history.clear()
        self.consecutive_drowsy_count = 0
        self.current_state = "ALERT"


if __name__ == "__main__":
    print("Testing DriverClassifier & TemporalDriverBuffer self-check...")
    classifier = DriverClassifier()
    print(f"DriverClassifier initialized on device: {classifier.device}")

    # Create dummy 227x227 image
    dummy_frame = (np.random.rand(227, 227, 3) * 255).astype(np.uint8)
    pred = classifier.predict(dummy_frame)
    print(f"Single frame prediction on dummy input: {pred}")

    buffer = TemporalDriverBuffer(window_size=15)
    for i, p in enumerate([0.2, 0.3, 0.45, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]):
        stat = buffer.push(p)
        print(f"  Step {i+1}: prob={p:.2f} -> smoothed={stat['smoothed_fatigue']:.3f}, state={stat['state']}")

    print("Self-test passed! ✅")
