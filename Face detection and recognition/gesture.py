import cv2
import numpy as np


class FingerCounter:
    def __init__(self):
        self.lower_hsv = np.array([0, 15, 60], dtype=np.uint8)
        self.upper_hsv = np.array([20, 150, 255], dtype=np.uint8)
        self.lower_ycrcb = np.array([0, 133, 77], dtype=np.uint8)
        self.upper_ycrcb = np.array([255, 173, 127], dtype=np.uint8)

    def _create_skin_mask(self, img: np.ndarray) -> np.ndarray:
        blurred = cv2.GaussianBlur(img, (5, 5), 0)

        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        mask_hsv = cv2.inRange(hsv, self.lower_hsv, self.upper_hsv)

        ycrcb = cv2.cvtColor(blurred, cv2.COLOR_BGR2YCrCb)
        mask_ycrcb = cv2.inRange(ycrcb, self.lower_ycrcb, self.upper_ycrcb)

        mask = cv2.bitwise_and(mask_hsv, mask_ycrcb)
        mask = cv2.medianBlur(mask, 5)
        mask = cv2.erode(mask, None, iterations=1)
        mask = cv2.dilate(mask, None, iterations=2)
        return mask

    def count_fingers(self, img: np.ndarray, draw: bool = True) -> tuple[int, np.ndarray | None]:
        mask = self._create_skin_mask(img)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return 0, mask

        hand = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(hand)
        if area < 8000:
            return 0, mask

        hull_idx = cv2.convexHull(hand, returnPoints=False)
        if hull_idx is None or len(hull_idx) < 3:
            return 0, mask

        defects = cv2.convexityDefects(hand, hull_idx)
        if defects is None:
            return 0, mask

        if draw:
            cv2.drawContours(img, [hand], -1, (0, 255, 0), 2)
            hull_pts = cv2.convexHull(hand)
            cv2.drawContours(img, [hull_pts], -1, (255, 0, 0), 2)
            cx, cy = np.mean(hand, axis=0)[0].astype(int)
            cv2.circle(img, (cx, cy), 5, (0, 0, 255), -1)

        finger_count = 0
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            start = tuple(hand[s][0])
            end = tuple(hand[e][0])
            far = tuple(hand[f][0])

            a = np.linalg.norm(np.array(start) - np.array(end))
            b = np.linalg.norm(np.array(start) - np.array(far))
            c = np.linalg.norm(np.array(end) - np.array(far))
            angle = np.degrees(np.arccos(np.clip((b**2 + c**2 - a**2) / max(2 * b * c, 1e-6), -1, 1)))

            if angle <= 90 and d > 15000:
                finger_count += 1
                if draw:
                    cv2.circle(img, far, 6, (0, 0, 255), -1)

        finger_count = min(finger_count + 1, 5)

        if draw:
            overlay = img.copy()
            cv2.rectangle(overlay, (5, 45), (195, 105), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
            cv2.putText(img, f"Fingers: {finger_count}", (15, 88),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

        return finger_count, mask
