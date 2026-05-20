import base64
import math
import threading
import time
from collections import deque
from pathlib import Path
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO


PHONE_CLASS_ID = 67
PHONE_CONFIDENCE = 0.30
PHONE_DETECT_EVERY_N_FRAMES = 3
PHONE_HOLD_TIME = 2.5
YOLO_PROCESS_WIDTH = 320

EAR_THRESHOLD = 0.21
DROWSY_SECONDS = 2.0
HEAD_DOWN_THRESHOLD = 0.65

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

FACE_MATCH_TOLERANCE = 0.5
FACE_RECOGNITION_WIDTH = 280
DRIVER_MATCH_EVERY_N_FRAMES = 12
DRIVER_HOLD_TIME = 1.0
PHONE_NEAR_FACE_MARGIN = 0.8


def decode_base64_image(image_base64: str) -> np.ndarray:
    if "," in image_base64:
        image_base64 = image_base64.split(",", 1)[1]

    image_bytes = base64.b64decode(image_base64)
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if frame is None:
        raise ValueError("Could not decode image data.")

    return frame


def decode_uploaded_file(image_bytes: bytes) -> np.ndarray:
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if frame is None:
        raise ValueError("Could not decode uploaded file.")

    return frame


def resize_frame_for_width(frame: np.ndarray, target_width: int) -> tuple[np.ndarray, float]:
    height, width = frame.shape[:2]
    if width <= target_width:
        return frame, 1.0

    scale = width / target_width
    target_height = int(height / scale)
    resized = cv2.resize(frame, (target_width, target_height))
    return resized, scale


def scale_xyxy_box(box: tuple[int, int, int, int], scale: float) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    return (
        int(x1 * scale),
        int(y1 * scale),
        int(x2 * scale),
        int(y2 * scale),
    )


def face_location_to_xyxy(face_location: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    top, right, bottom, left = face_location
    return left, top, right, bottom


def expand_xyxy_box(box: tuple[int, int, int, int], margin_ratio: float) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    width = x2 - x1
    height = y2 - y1
    margin_x = int(width * margin_ratio)
    margin_y = int(height * margin_ratio)
    return (
        x1 - margin_x,
        y1 - margin_y,
        x2 + margin_x,
        y2 + margin_y,
    )


def boxes_intersect(
    box_a: tuple[int, int, int, int],
    box_b: tuple[int, int, int, int],
) -> bool:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    return ax1 < bx2 and ax2 > bx1 and ay1 < by2 and ay2 > by1


def clamp_xyxy_box(box: tuple[int, int, int, int], frame_shape: tuple[int, ...]) -> tuple[int, int, int, int]:
    frame_height, frame_width = frame_shape[:2]
    x1, y1, x2, y2 = box
    return (
        max(0, x1),
        max(0, y1),
        min(frame_width, x2),
        min(frame_height, y2),
    )


def calculate_ear(
    eye_points: list[int],
    landmarks,
    width: int,
    height: int,
    offset: tuple[int, int] = (0, 0),
) -> tuple[float, list[tuple[int, int]]]:
    offset_x, offset_y = offset
    coords = [
        (
            int(landmarks[point].x * width) + offset_x,
            int(landmarks[point].y * height) + offset_y,
        )
        for point in eye_points
    ]

    vertical_1 = math.dist(coords[1], coords[5])
    vertical_2 = math.dist(coords[2], coords[4])
    horizontal = math.dist(coords[0], coords[3])

    if horizontal == 0:
        return 0.0, coords

    ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
    return ear, coords


def is_head_forward(landmarks) -> bool:
    eye_y = (landmarks[33].y + landmarks[263].y) / 2.0
    nose_y = landmarks[1].y
    mouth_y = landmarks[13].y

    eye_to_mouth = mouth_y - eye_y
    if eye_to_mouth <= 0:
        return True

    down_score = (nose_y - eye_y) / eye_to_mouth
    return down_score <= HEAD_DOWN_THRESHOLD


class WebDrowsinessDetector:
    def __init__(self, model_path: str | Path):
        self.model_path = str(model_path)
        self.face_recognition = self._load_face_recognition()
        self.yolo_model, self.face_mesh = self._init_models()
        self.state = self._create_state()
        self.lock = threading.Lock()
        self.started_at = time.time()
        self.history = deque(maxlen=180)

    def _load_face_recognition(self):
        import face_recognition

        return face_recognition

    def _init_models(self):
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        yolo_model = YOLO(self.model_path)
        return yolo_model, face_mesh

    def _create_state(self) -> dict:
        return {
            "driver_encoding": None,
            "frame_count": 0,
            "driver_face_location": None,
            "driver_match_distance": None,
            "last_driver_seen_time": 0.0,
            "phone_detected": False,
            "phone_box": None,
            "phone_confidence": 0.0,
            "last_phone_seen_time": 0.0,
            "eyes_closed_since": None,
            "status": "INITIALIZING",
            "mode": "ENROLLMENT",
            "baseline_ear": None,
            "baseline_samples": [],
            "calibrated": False,
            "fatigue_score": 0,
            "attention_score": 100,
            "alert": None,
            "total_frames": 0,
            "drowsy_frames": 0,
            "phone_frames": 0,
            "driver_missing_frames": 0,
            "last_prediction": None,
        }

    # OLD CODE (BACKUP)
    # The desktop version used a cv2.VideoCapture loop and live window drawing.
    # For the web app we remove the webcam loop and process one frame per API call.
    #
    # while True:
    #     ret, frame = cap.read()
    #     run_phone_detection(frame, model, app_state)
    #     driver_location, match_distance = update_driver_match(frame, driver_encoding, app_state)
    #     ear, left_coords, right_coords, head_forward = run_driver_face_detection(...)
    #     handle_drowsiness(frame, ear, app_state, args.ear_threshold, suppress_reason=...)
    #     cv2.imshow(...)

    # NEW CODE (WEB API)
    def process_frame(self, frame: np.ndarray) -> dict:
        with self.lock:
            return self._process_frame_locked(frame)

    def _process_frame_locked(self, frame: np.ndarray) -> dict:
        self.state["frame_count"] += 1
        self.state["total_frames"] += 1
        frame = self._normalize_frame(frame)

        if self.state["driver_encoding"] is None:
            result = self._enroll_driver(frame)
            self._record_prediction(result)
            return result

        self._run_phone_detection(frame)
        driver_location, match_distance = self._update_driver_match(frame)

        ear = None
        head_forward = True
        driver_found = driver_location is not None

        if driver_found:
            ear, _, _, head_forward = self.run_driver_face_detection(frame, driver_location)

        if driver_found and ear is not None:
            self._update_baseline(ear)
            suppress_reason = self._get_drowsiness_suppression_reason(
                driver_location,
                head_forward,
                ear,
            )
            self.handle_drowsiness(ear, suppress_reason=suppress_reason)
        else:
            self.state["eyes_closed_since"] = None
            self.state["status"] = "DRIVER NOT FOUND"

        self._update_mode()
        self._update_scores(ear=ear, driver_found=driver_found)
        result = {
            "status": self.state["status"],
            "ear": round(float(ear), 4) if ear is not None else None,
            "baseline_ear": self._rounded_baseline(),
            "mode": self.state["mode"],
            "fatigue_score": self.state["fatigue_score"],
            "attention_score": self.state["attention_score"],
            "phone_detected": bool(self.state["phone_detected"]),
            "driver_found": bool(driver_found),
            "calibrated": bool(self.state["calibrated"]),
            "alert": self.state["alert"],
        }
        self._record_prediction(result)
        return result

    def _normalize_frame(self, frame: np.ndarray) -> np.ndarray:
        if frame.ndim != 3:
            raise ValueError("Expected a color image.")
        return np.ascontiguousarray(frame, dtype=np.uint8)

    def _enroll_driver(self, frame: np.ndarray) -> dict:
        rgb_frame, face_locations, _ = self.get_face_locations(frame)
        driver_location = self.get_largest_face_location(face_locations)

        if driver_location is None:
            self.state["status"] = "SHOW FACE TO CAMERA"
            self.state["mode"] = "ENROLLMENT"
            return {
                "status": self.state["status"],
                "ear": None,
                "baseline_ear": self._rounded_baseline(),
                "mode": self.state["mode"],
                "fatigue_score": self.state["fatigue_score"],
                "attention_score": self.state["attention_score"],
                "phone_detected": False,
                "driver_found": False,
                "calibrated": bool(self.state["calibrated"]),
                "alert": "Position the driver in frame",
            }

        driver_encoding = self.extract_face_encoding(rgb_frame, driver_location)
        if driver_encoding is None:
            self.state["status"] = "FACE NOT CLEAR"
            self.state["mode"] = "ENROLLMENT"
            return {
                "status": self.state["status"],
                "ear": None,
                "baseline_ear": self._rounded_baseline(),
                "mode": self.state["mode"],
                "fatigue_score": self.state["fatigue_score"],
                "attention_score": self.state["attention_score"],
                "phone_detected": False,
                "driver_found": False,
                "calibrated": bool(self.state["calibrated"]),
                "alert": "Face is not clear enough",
            }

        self.state["driver_encoding"] = driver_encoding
        self.state["driver_face_location"] = driver_location
        self.state["driver_match_distance"] = 0.0
        self.state["last_driver_seen_time"] = time.time()
        self.state["status"] = "DRIVER ENROLLED"
        self.state["mode"] = "ENROLLMENT"
        return {
            "status": self.state["status"],
            "ear": None,
            "baseline_ear": self._rounded_baseline(),
            "mode": self.state["mode"],
            "fatigue_score": self.state["fatigue_score"],
            "attention_score": self.state["attention_score"],
            "phone_detected": False,
            "driver_found": True,
            "calibrated": bool(self.state["calibrated"]),
            "alert": "Driver enrolled",
        }

    def get_face_locations(self, frame: np.ndarray, process_width: Optional[int] = None):
        process_frame, scale = resize_frame_for_width(frame, process_width or frame.shape[1])
        rgb_frame = cv2.cvtColor(process_frame, cv2.COLOR_BGR2RGB)
        rgb_frame = np.ascontiguousarray(rgb_frame, dtype=np.uint8)
        face_locations = self.face_recognition.face_locations(rgb_frame)
        return rgb_frame, face_locations, scale

    def get_largest_face_location(self, face_locations):
        if not face_locations:
            return None
        return max(face_locations, key=self.face_area)

    def face_area(self, face_location) -> int:
        top, right, bottom, left = face_location
        return (right - left) * (bottom - top)

    def extract_face_encoding(self, rgb_frame: np.ndarray, face_location):
        encodings = self.face_recognition.face_encodings(rgb_frame, [face_location])
        if not encodings:
            return None
        return encodings[0]

    def compare_face_encoding(self, driver_encoding, face_encoding, tolerance=FACE_MATCH_TOLERANCE):
        matches = self.face_recognition.compare_faces(
            [driver_encoding],
            face_encoding,
            tolerance=tolerance,
        )
        distance = self.face_recognition.face_distance([driver_encoding], face_encoding)[0]
        return matches[0], distance

    # OLD CODE (BACKUP)
    # This re-ran face recognition on every frame. That was too expensive for a
    # browser client that sends frequent API requests.
    #
    # def find_driver_face(self, frame):
    #     rgb_frame, face_locations, scale = self.get_face_locations(frame)
    #     ...

    # NEW CODE (OPTIMIZED)
    def find_driver_face(self, frame: np.ndarray):
        rgb_frame, face_locations, scale = self.get_face_locations(
            frame,
            process_width=FACE_RECOGNITION_WIDTH,
        )
        best_match = None

        for face_location in face_locations:
            face_encoding = self.extract_face_encoding(rgb_frame, face_location)
            if face_encoding is None:
                continue

            is_match, distance = self.compare_face_encoding(
                self.state["driver_encoding"],
                face_encoding,
            )

            if is_match and (best_match is None or distance < best_match[1]):
                best_match = (self.scale_face_location(face_location, scale), distance)

        if best_match is None:
            return None, None

        return best_match

    def scale_face_location(self, face_location, scale: float):
        top, right, bottom, left = face_location
        return (
            int(top * scale),
            int(right * scale),
            int(bottom * scale),
            int(left * scale),
        )

    def _should_refresh_driver_match(self) -> bool:
        if self.state["driver_face_location"] is None:
            return True

        current_time = time.time()
        recently_seen = current_time - self.state["last_driver_seen_time"] < DRIVER_HOLD_TIME
        frame_due = self.state["frame_count"] % DRIVER_MATCH_EVERY_N_FRAMES == 0
        return frame_due or not recently_seen

    def _update_driver_match(self, frame: np.ndarray):
        if self._should_refresh_driver_match():
            driver_location, match_distance = self.find_driver_face(frame)
            if driver_location is not None:
                self.state["driver_face_location"] = driver_location
                self.state["driver_match_distance"] = match_distance
                self.state["last_driver_seen_time"] = time.time()

        if time.time() - self.state["last_driver_seen_time"] < DRIVER_HOLD_TIME:
            return self.state["driver_face_location"], self.state["driver_match_distance"]

        self.state["driver_face_location"] = None
        self.state["driver_match_distance"] = None
        return None, None

    # OLD CODE (BACKUP)
    # The desktop version drew boxes and beeped. The API version only updates state.
    #
    # def run_phone_detection(frame, model, app_state):
    #     results = model(frame)
    #     ...

    # NEW CODE (OPTIMIZED)
    def _run_phone_detection(self, frame: np.ndarray) -> None:
        if self.state["frame_count"] % PHONE_DETECT_EVERY_N_FRAMES == 0:
            yolo_frame, scale = resize_frame_for_width(frame, YOLO_PROCESS_WIDTH)
            results = self.yolo_model(yolo_frame, verbose=False)
            best_phone = None

            for result in results:
                for box in result.boxes:
                    cls = int(box.cls[0])
                    confidence = float(box.conf[0])
                    if cls == PHONE_CLASS_ID and confidence >= PHONE_CONFIDENCE:
                        if best_phone is None or confidence > best_phone[0]:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            best_phone = (confidence, scale_xyxy_box((x1, y1, x2, y2), scale))

            if best_phone is not None:
                self.state["last_phone_seen_time"] = time.time()
                self.state["phone_confidence"] = best_phone[0]
                self.state["phone_box"] = best_phone[1]

        if time.time() - self.state["last_phone_seen_time"] < PHONE_HOLD_TIME:
            self.state["phone_detected"] = True
        else:
            self.state["phone_detected"] = False
            self.state["phone_box"] = None
            self.state["phone_confidence"] = 0.0

    def run_driver_face_detection(self, frame: np.ndarray, driver_face_location):
        face_box = face_location_to_xyxy(driver_face_location)
        face_box = expand_xyxy_box(face_box, margin_ratio=0.25)
        left, top, right, bottom = clamp_xyxy_box(face_box, frame.shape)
        face_roi = frame[top:bottom, left:right]

        if face_roi.size == 0:
            return None, None, None, True

        roi_height, roi_width, _ = face_roi.shape
        rgb_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
        rgb_roi = np.ascontiguousarray(rgb_roi, dtype=np.uint8)
        result = self.face_mesh.process(rgb_roi)

        if not result.multi_face_landmarks:
            return None, None, None, True

        landmarks = result.multi_face_landmarks[0].landmark
        head_forward = is_head_forward(landmarks)
        offset = (left, top)

        left_ear, left_coords = calculate_ear(
            LEFT_EYE,
            landmarks,
            roi_width,
            roi_height,
            offset=offset,
        )
        right_ear, right_coords = calculate_ear(
            RIGHT_EYE,
            landmarks,
            roi_width,
            roi_height,
            offset=offset,
        )
        ear = (left_ear + right_ear) / 2.0
        return ear, left_coords, right_coords, head_forward

    def _update_baseline(self, ear: float) -> None:
        if ear is None:
            return

        if not self.state["calibrated"] and self.state["status"] == "FOCUSED":
            self.state["baseline_samples"].append(ear)
            if len(self.state["baseline_samples"]) >= 15:
                self.state["baseline_ear"] = sum(self.state["baseline_samples"]) / len(
                    self.state["baseline_samples"]
                )
                self.state["calibrated"] = True

    # OLD CODE (BACKUP)
    # The desktop version only checked EAR. Looking down at a nearby phone could
    # create a false drowsy alert.
    #
    # if ear < EAR_THRESHOLD:
    #     status = "DROWSY"

    # NEW CODE (OPTIMIZED)
    def handle_drowsiness(self, ear: float, suppress_reason: Optional[str] = None) -> None:
        if suppress_reason is not None:
            self.state["eyes_closed_since"] = None
            self.state["status"] = suppress_reason
            self.state["alert"] = "Distraction detected"
            return

        current_time = time.time()
        if self.state["calibrated"] and self.state["baseline_ear"] is not None:
            threshold = self.state["baseline_ear"] * 0.75
        else:
            threshold = EAR_THRESHOLD

        if ear < threshold:
            if self.state["eyes_closed_since"] is None:
                self.state["eyes_closed_since"] = current_time

            if current_time - self.state["eyes_closed_since"] >= DROWSY_SECONDS:
                self.state["status"] = "DROWSY"
                self.state["alert"] = "Drowsiness risk"
            else:
                self.state["status"] = "EYES CLOSING"
                self.state["alert"] = "Eyes closing"
        else:
            self.state["eyes_closed_since"] = None
            self.state["status"] = "FOCUSED"
            self.state["alert"] = None

    def _get_drowsiness_suppression_reason(
        self,
        driver_face_location,
        head_forward: bool,
        ear: float,
    ) -> Optional[str]:
        if ear is not None and ear < EAR_THRESHOLD * 0.8:
            return None

        if not head_forward:
            return "LOOKING DOWN"

        if self.state["phone_detected"] and self._is_phone_near_face(
            self.state["phone_box"],
            driver_face_location,
        ):
            return "PHONE DETECTED"

        return None

    def _is_phone_near_face(self, phone_box, face_location) -> bool:
        if phone_box is None or face_location is None:
            return False

        expanded_face_box = expand_xyxy_box(
            face_location_to_xyxy(face_location),
            PHONE_NEAR_FACE_MARGIN,
        )
        return boxes_intersect(phone_box, expanded_face_box)

    def _update_mode(self) -> None:
        status = self.state["status"]
        if status == "DROWSY":
            self.state["mode"] = "DROWSY MODE"
        elif status in {"PHONE DETECTED", "LOOKING DOWN"}:
            self.state["mode"] = "DISTRACTION MODE"
        elif status == "FOCUSED":
            self.state["mode"] = "FOCUSED MODE"
        elif status == "DRIVER NOT FOUND":
            self.state["mode"] = "NO DRIVER"
        elif status in {"DRIVER ENROLLED", "SHOW FACE TO CAMERA", "FACE NOT CLEAR"}:
            self.state["mode"] = "ENROLLMENT"
        else:
            self.state["mode"] = "MONITORING"

    def _update_scores(self, ear: Optional[float], driver_found: bool) -> None:
        if self.state["status"] == "DROWSY":
            self.state["drowsy_frames"] += 1
        if self.state["phone_detected"]:
            self.state["phone_frames"] += 1
        if not driver_found:
            self.state["driver_missing_frames"] += 1
            self.state["alert"] = "Driver not found"

        if ear is None:
            ear_pressure = 35
        else:
            baseline = self.state["baseline_ear"] or max(EAR_THRESHOLD, ear)
            ratio = min(1.0, max(0.0, 1.0 - (ear / max(baseline, 0.01))))
            ear_pressure = int(ratio * 100)

        fatigue_score = ear_pressure
        if self.state["status"] == "DROWSY":
            fatigue_score = max(fatigue_score, 92)
        elif self.state["status"] == "EYES CLOSING":
            fatigue_score = max(fatigue_score, 68)

        attention_score = 100 - int(fatigue_score * 0.55)
        if self.state["phone_detected"]:
            attention_score -= 25
        if self.state["status"] == "LOOKING DOWN":
            attention_score -= 20
        if not driver_found:
            attention_score = 0

        self.state["fatigue_score"] = int(min(100, max(0, fatigue_score)))
        self.state["attention_score"] = int(min(100, max(0, attention_score)))

    def _rounded_baseline(self) -> Optional[float]:
        baseline = self.state.get("baseline_ear")
        return round(float(baseline), 4) if baseline is not None else None

    def _record_prediction(self, result: dict) -> None:
        result = dict(result)
        result["timestamp"] = time.time()
        self.state["last_prediction"] = result
        self.history.append(result)

    def get_session_stats(self) -> dict:
        with self.lock:
            return self._session_stats_unlocked()

    def reset_session(self) -> dict:
        with self.lock:
            self.state = self._create_state()
            self.started_at = time.time()
            self.history.clear()
            return self._session_stats_unlocked()

    def _session_stats_unlocked(self) -> dict:
        total_frames = max(1, int(self.state["total_frames"]))
        elapsed = max(0.0, time.time() - self.started_at)
        return {
            "session_seconds": int(elapsed),
            "frames_processed": int(self.state["total_frames"]),
            "drowsy_events": int(self.state["drowsy_frames"]),
            "phone_events": int(self.state["phone_frames"]),
            "driver_missing_events": int(self.state["driver_missing_frames"]),
            "drowsy_ratio": round(self.state["drowsy_frames"] / total_frames, 4),
            "phone_ratio": round(self.state["phone_frames"] / total_frames, 4),
            "calibrated": bool(self.state["calibrated"]),
            "baseline_ear": self._rounded_baseline(),
            "latest": self.state["last_prediction"],
            "history": list(self.history)[-60:],
        }
