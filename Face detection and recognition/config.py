import os
from pathlib import Path
import cv2

BASE_DIR = Path(__file__).parent

DATA_DIR = BASE_DIR / "data"
KNOWN_FACES_DIR = DATA_DIR / "known_faces"
TEST_IMAGES_DIR = DATA_DIR / "test_images"
TEST_VIDEOS_DIR = DATA_DIR / "test_videos"
OUTPUT_DIR = DATA_DIR / "output"

MODELS_DIR = BASE_DIR / "models"

for d in [DATA_DIR, KNOWN_FACES_DIR, TEST_IMAGES_DIR, TEST_VIDEOS_DIR, OUTPUT_DIR, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

HAAR_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

DNN_PROTOTXT = str(MODELS_DIR / "deploy.prototxt")
DNN_MODEL = str(MODELS_DIR / "res10_300x300_ssd_iter_140000.caffemodel")
DNN_CONFIDENCE_THRESHOLD = 0.5

RECOGNITION_METHODS = ["lbph", "eigen", "fisher"]
DEFAULT_RECOGNITION_METHOD = "lbph"
DEFAULT_DETECTION_METHOD = "haar"
FACE_RECOGNITION_TOLERANCE = 0.6
