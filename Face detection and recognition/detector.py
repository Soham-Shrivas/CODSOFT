import cv2
import numpy as np
from pathlib import Path
import config
import urllib.request


class FaceDetector:
    def __init__(self, method: str = "haar"):
        self.method = method.lower()
        self._init_haar()
        self._init_dnn()

    def _init_haar(self):
        cascade_path = config.HAAR_CASCADE_PATH
        if not Path(cascade_path).exists():
            raise FileNotFoundError(
                f"Haar cascade not found at {cascade_path}. "
                "OpenCV data files may be missing."
            )
        self.haar_cascade = cv2.CascadeClassifier(cascade_path)

    def _init_dnn(self):
        self.dnn_net = None
        proto = Path(config.DNN_PROTOTXT)
        model = Path(config.DNN_MODEL)
        if proto.exists() and model.exists():
            self.dnn_net = cv2.dnn.readNetFromCaffe(
                str(proto), str(model)
            )
        else:
            print("DNN model files not found. Run download_dnn_model() to fetch them.")

    def detect_haar(self, img: np.ndarray) -> list[dict]:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.haar_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        results = []
        for (x, y, w, h) in faces:
            results.append({
                "bbox": (x, y, w, h),
                "confidence": None,
                "method": "haar"
            })
        return results

    def detect_dnn(self, img: np.ndarray) -> list[dict]:
        if self.dnn_net is None:
            print("DNN detector not available, falling back to Haar.")
            return self.detect_haar(img)

        h, w = img.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(img, (300, 300)), 1.0,
            (300, 300), (104.0, 177.0, 123.0)
        )
        self.dnn_net.setInput(blob)
        detections = self.dnn_net.forward()

        results = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence < config.DNN_CONFIDENCE_THRESHOLD:
                continue
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)
            results.append({
                "bbox": (x1, y1, x2 - x1, y2 - y1),
                "confidence": float(confidence),
                "method": "dnn"
            })
        return results

    def detect(self, img: np.ndarray) -> list[dict]:
        if self.method == "dnn":
            return self.detect_dnn(img)
        return self.detect_haar(img)


def download_dnn_model():
    proto_url = (
        "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
    )
    model_url = (
        "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/"
        "res10_300x300_ssd_iter_140000.caffemodel"
    )
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    proto_path = config.MODELS_DIR / "deploy.prototxt"
    model_path = config.MODELS_DIR / "res10_300x300_ssd_iter_140000.caffemodel"

    if not proto_path.exists():
        print("Downloading DNN prototxt...")
        urllib.request.urlretrieve(proto_url, str(proto_path))
    if not model_path.exists():
        print("Downloading DNN model (this may take a while)...")
        urllib.request.urlretrieve(model_url, str(model_path))
    print("DNN model files ready.")
