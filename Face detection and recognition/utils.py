import cv2
import numpy as np
from pathlib import Path
import config


def load_image(path: str) -> np.ndarray | None:
    img = cv2.imread(str(path))
    if img is None:
        print(f"Could not load image: {path}")
        return None
    return img


def save_image(path: str, img: np.ndarray) -> bool:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return cv2.imwrite(str(path), img)


def resize_to_fit(img: np.ndarray, max_width: int = 1280, max_height: int = 720) -> np.ndarray:
    h, w = img.shape[:2]
    scale = min(max_width / w, max_height / h, 1.0)
    if scale < 1.0:
        new_w = int(w * scale)
        new_h = int(h * scale)
        return cv2.resize(img, (new_w, new_h))
    return img


def draw_face_box(img: np.ndarray, x: int, y: int, w: int, h: int,
                  label: str = "", confidence: float = None, color: tuple = (0, 255, 0)):
    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
    if label or confidence is not None:
        text_parts = []
        if label:
            text_parts.append(label)
        if confidence is not None:
            text_parts.append(f"{confidence:.2f}")
        text = " - ".join(text_parts)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x, y - th - 10), (x + tw + 10, y), color, -1)
        cv2.putText(img, text, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


def list_known_faces() -> list[Path]:
    extensions = ("*.jpg", "*.jpeg", "*.png", "*.pgm")
    faces = []
    for ext in extensions:
        faces.extend(config.KNOWN_FACES_DIR.glob(ext))
        faces.extend(config.KNOWN_FACES_DIR.glob(ext.upper()))
    return sorted(faces)


def get_label_from_filename(path: Path) -> str:
    return path.stem.split("_")[0] if "_" in path.stem else path.stem
