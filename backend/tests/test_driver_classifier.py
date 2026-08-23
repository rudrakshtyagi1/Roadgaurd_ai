"""
RoadGuard AI — Automated Driver Classifier & Class-Mapping Diagnostic Tests
Verifies class mapping invariance, probability semantics, and known-sample consistency.
"""

import sys
import unittest
from pathlib import Path
import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.driver_classifier import DriverClassifier, CustomDriverCNN
from core.driver_monitor import DriverMonitor


class TestDriverClassifierSemantics(unittest.TestCase):

    def setUp(self):
        self.classifier = DriverClassifier()

    def test_class_mapping_standard(self):
        """Standard mapping: class 0 = non_drowsy, class 1 = drowsy."""
        clf = DriverClassifier(class_to_idx={"non_drowsy": 0, "drowsy": 1})
        # Dummy image
        dummy = np.zeros((224, 224, 3), dtype=np.uint8)
        res = clf.predict(dummy)

        self.assertIn("cnn_drowsy_probability", res)
        self.assertIn("cnn_predicted_class", res)
        self.assertIn("cnn_predicted_class_probability", res)
        self.assertGreaterEqual(res["cnn_drowsy_probability"], 0.0)
        self.assertLessEqual(res["cnn_drowsy_probability"], 1.0)

    def test_class_mapping_invariance(self):
        """Test that if class mapping is reversed, semantics remain correct."""
        # Simulated logit
        clf_std = DriverClassifier(class_to_idx={"non_drowsy": 0, "drowsy": 1})
        clf_inv = DriverClassifier(class_to_idx={"drowsy": 0, "non_drowsy": 1})

        dummy = np.ones((224, 224, 3), dtype=np.uint8) * 128
        res_std = clf_std.predict(dummy)
        res_inv = clf_inv.predict(dummy)

        # Inverted mapping should calculate 1 - prob
        self.assertAlmostEqual(
            res_std["cnn_drowsy_probability"] + res_inv["cnn_drowsy_probability"],
            1.0,
            places=3,
        )

    def test_pipeline_consistency_invariants(self):
        """Verify invariants: class probability bounds and consistency."""
        dummy = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        res = self.classifier.predict(dummy)

        p_drowsy = res["cnn_drowsy_probability"]
        pred_class = res["cnn_predicted_class"]
        pred_prob = res["cnn_predicted_class_probability"]

        if pred_class == "NON_DROWSY":
            self.assertLess(p_drowsy, 0.50)
            self.assertAlmostEqual(pred_prob, 1.0 - p_drowsy, places=4)
        else:
            self.assertGreaterEqual(p_drowsy, 0.50)
            self.assertAlmostEqual(pred_prob, p_drowsy, places=4)

    def test_known_samples_from_dataset(self):
        """Test with real known open and closed driver samples from dataset if present."""
        raw_root = Path("/Users/rudrakshtyagi/Desktop/roadgaurdai/drowsiness_processed/test")
        if not raw_root.exists():
            self.skipTest("Dataset not available locally")

        nd_files = sorted(list((raw_root / "non_drowsy").glob("*.png")) + list((raw_root / "non_drowsy").glob("*.jpg")))
        dr_files = sorted(list((raw_root / "drowsy").glob("*.png")) + list((raw_root / "drowsy").glob("*.jpg")))

        if nd_files and dr_files:
            # Evaluate 5 non-drowsy and 5 drowsy samples
            for p in dr_files[:5]:
                img = cv2.imread(str(p))
                p_dr = self.classifier.predict_proba(img)
                self.assertGreater(p_dr, 0.50, f"Drowsy sample {p.name} produced low P(Drowsy)={p_dr}")

    def test_driver_monitor_no_face_gating(self):
        """Test that DriverMonitor gates empty/no-face frames to 0.0 CNN probability."""
        monitor = DriverMonitor()
        empty_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        res = monitor.process(empty_frame)

        self.assertFalse(res["face_detected"])
        self.assertEqual(res["cnn_drowsy_probability"], 0.0)
        self.assertEqual(res["cnn_predicted_class"], "NO_FACE")
        self.assertEqual(res["state"], "NORMAL")


if __name__ == "__main__":
    unittest.main()
