#!/usr/bin/env python3
"""
Face Detection & Recognition Application

Usage:
    python main.py --mode image --path <image_path>
    python main.py --mode video --path <video_path>
    python main.py --mode webcam
    python main.py --mode train
    python main.py --download-dnn

Examples:
    python main.py --mode webcam --detect dnn
    python main.py --mode webcam --gesture
    python main.py --mode webcam --detect dnn --recognize --gesture
    python main.py --mode image --path data/test_images/group.jpg --recognize
"""

import argparse
import sys
import cv2
import numpy as np
from pathlib import Path

import config
import utils
from detector import FaceDetector, download_dnn_model
from recognizer import FaceRecognizer
from gesture import FingerCounter


def process_image(detector, img, recognizer=None, draw=True, finger_counter=None):
    results = detector.detect(img)
    if draw:
        display = img.copy()
        for r in results:
            x, y, w, h = r["bbox"]
            label = ""
            conf = r["confidence"]

            if recognizer and recognizer.is_trained():
                face_roi = img[y:y+h, x:x+w]
                if face_roi.size > 0:
                    name, rec_conf = recognizer.recognize(face_roi)
                    label = name if name else "Unknown"

            utils.draw_face_box(display, x, y, w, h, label=label, confidence=conf)

        if finger_counter:
            finger_count, mask = finger_counter.count_fingers(display, draw=True)
            cv2.putText(display, f"Fingers: {finger_count}", (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        return display, results
    return img, results


def process_video(detector, video_path: str, recognizer=None, show=True, output_path: str = None, finger_counter=None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Could not open video: {video_path}")
        return

    writer = None
    if output_path:
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        processed, results = process_image(detector, frame, recognizer, finger_counter=finger_counter)
        frame_count += 1

        if writer:
            writer.write(processed)
        if show:
            info = f"Frame: {frame_count} | Faces: {len(results)}"
            cv2.putText(processed, info, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow("Face Detection - Press ESC to exit", processed)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print(f"Processed {frame_count} frames. Found faces in most frames.")


def webcam_loop(detector, recognizer=None, finger_counter=None):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        return

    print("Webcam started. Press ESC to quit, SPACE to capture snapshot.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        processed, results = process_image(detector, frame, recognizer, finger_counter=finger_counter)

        info = f"Faces: {len(results)} | Detector: {detector.method.upper()}"
        cv2.putText(processed, info, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("Face Detection - Webcam", processed)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        elif key == ord(" "):
            ts = cv2.getTickCount()
            snap_path = config.OUTPUT_DIR / f"snapshot_{int(ts)}.jpg"
            cv2.imwrite(str(snap_path), processed)
            print(f"Snapshot saved: {snap_path}")

    cap.release()
    cv2.destroyAllWindows()


def train_recognizer(method: str = "lbph"):
    recognizer = FaceRecognizer(method=method)
    success = recognizer.train_from_dir()
    if success:
        model_path = config.OUTPUT_DIR / f"{method}_model.yml"
        recognizer.model.save(str(model_path))
        print(f"Model saved to {model_path}")
    return recognizer if success else None


def main():
    parser = argparse.ArgumentParser(
        description="Face Detection & Recognition",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("--mode", choices=["image", "video", "webcam", "train"],
                        default="webcam", help="Operation mode")
    parser.add_argument("--path", type=str, help="Path to image or video file")
    parser.add_argument("--detect", choices=["haar", "dnn"], default="haar",
                        help="Detection method (default: haar)")
    parser.add_argument("--recognize", action="store_true",
                        help="Enable face recognition")
    parser.add_argument("--rec-method", choices=config.RECOGNITION_METHODS,
                        default=config.DEFAULT_RECOGNITION_METHOD,
                        help="Recognition algorithm")
    parser.add_argument("--output", type=str, help="Output path for processed file")
    parser.add_argument("--download-dnn", action="store_true",
                        help="Download DNN model files and exit")
    parser.add_argument("--no-display", action="store_true",
                        help="Disable display window (headless)")
    parser.add_argument("--save", action="store_true",
                        help="Save output to data/output/")
    parser.add_argument("--gesture", action="store_true",
                        help="Enable finger counting using hand gesture detection")

    args = parser.parse_args()

    if args.download_dnn:
        download_dnn_model()
        return

    recognizer = None
    if args.recognize or args.mode == "train":
        print(f"Initializing face recognizer ({args.rec_method})...")
        recognizer = FaceRecognizer(method=args.rec_method)
        if args.mode == "train":
            recognizer.train_from_dir()
            model_path = config.OUTPUT_DIR / f"{args.rec_method}_model.yml"
            recognizer.model.save(str(model_path))
            print(f"Training complete. Model saved to {model_path}")
            return
        trained = recognizer.train_from_dir()
        if not trained:
            print("Warning: No known faces found. Recognition disabled.")
            recognizer = None

    print(f"Initializing face detector ({args.detect})...")
    detector = FaceDetector(method=args.detect)

    if args.mode == "image":
        path = args.path
        if not path:
            print("Error: --path required for image mode.")
            sys.exit(1)
        img = utils.load_image(path)
        if img is None:
            sys.exit(1)
        display, results = process_image(detector, img, recognizer)
        print(f"Detected {len(results)} face(s) in image.")
        if args.save or args.output:
            out = args.output or str(config.OUTPUT_DIR / f"output_{Path(path).name}")
            utils.save_image(out, display)
            print(f"Saved: {out}")
        if not args.no_display:
            display = utils.resize_to_fit(display)
            cv2.imshow("Face Detection Result", display)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    elif args.mode == "video":
        path = args.path
        if not path:
            print("Error: --path required for video mode.")
            sys.exit(1)
        if not Path(path).exists():
            print(f"File not found: {path}")
            sys.exit(1)
        out_path = args.output
        if args.save and not out_path:
            out_path = str(config.OUTPUT_DIR / f"output_{Path(path).name}")
        process_video(detector, path, recognizer, show=not args.no_display,
                      output_path=out_path, finger_counter=finger_counter)

    finger_counter = None
    if args.gesture:
        print("Initializing finger counter...")
        finger_counter = FingerCounter()

    if args.mode == "webcam":
        webcam_loop(detector, recognizer, finger_counter)


if __name__ == "__main__":
    main()
