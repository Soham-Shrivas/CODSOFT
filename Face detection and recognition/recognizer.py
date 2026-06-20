import cv2
import numpy as np
from pathlib import Path
import config
import utils


class FaceRecognizer:
    def __init__(self, method: str = "lbph"):
        self.method = method.lower()
        self.model = self._create_model()
        self.labels: dict[int, str] = {}
        self.label_ids: dict[str, int] = {}
        self._trained = False

    def _create_model(self):
        if self.method == "eigen":
            return cv2.face.EigenFaceRecognizer_create()
        elif self.method == "fisher":
            return cv2.face.FisherFaceRecognizer_create()
        return cv2.face.LBPHFaceRecognizer_create()

    def train_from_dir(self, faces_dir: Path = None):
        if faces_dir is None:
            faces_dir = config.KNOWN_FACES_DIR

        image_paths = list(faces_dir.glob("*.*"))
        if not image_paths:
            print(f"No training images found in {faces_dir}")
            return False

        faces = []
        labels = []
        current_id = 0

        for img_path in image_paths:
            label = utils.get_label_from_filename(img_path)
            if label not in self.label_ids:
                self.label_ids[label] = current_id
                self.labels[current_id] = label
                current_id += 1

            img = utils.load_image(str(img_path))
            if img is None:
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (200, 200))
            faces.append(gray)
            labels.append(self.label_ids[label])

        if not faces:
            print("No valid faces found for training.")
            return False

        self.model.train(faces, np.array(labels))
        self._trained = True
        print(f"Trained on {len(faces)} samples across {len(self.labels)} subject(s): "
              f"{', '.join(self.labels.values())}")
        return True

    def recognize(self, face_roi: np.ndarray) -> tuple[str | None, float]:
        if not self._trained:
            return None, 0.0

        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (200, 200))

        label_id, confidence = self.model.predict(gray)
        name = self.labels.get(label_id, "Unknown")
        return name, confidence

    def is_trained(self) -> bool:
        return self._trained
