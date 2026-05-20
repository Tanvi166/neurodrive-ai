import math
import threading
import time

import cv2
import mediapipe as mp
import numpy as np
import winsound
from ultralytics import YOLO


PHONE_CLASS_ID = 67
PHONE_CONFIDENCE = 0.45
PHONE_DETECT_EVERY_N_FRAMES = 8
PHONE_HOLD_TIME = 0.8
PHONE_COOLDOWN = 0.8
YOLO_PROCESS_WIDTH = 320

EAR_THRESHOLD = 0.25
DROWSY_SECONDS = 1.2
DROWSY_COOLDOWN = 1.5
HEAD_DOWN_THRESHOLD = 0.65

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

WINDOW_NAME = "Drowsiness & Phone Detection"
FACE_MATCH_TOLERANCE = 0.5
FACE_RECOGNITION_WIDTH = 280
DRIVER_MATCH_EVERY_N_FRAMES = 12
DRIVER_HOLD_TIME = 1.0
PHONE_NEAR_FACE_MARGIN = 0.8




def play_beep(frequency, duration_ms):
    threading.Thread(
        target=winsound.Beep,
        args=(frequency, duration_ms),
        daemon=True,
    ).start()


def init_models(model_path):
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    model = YOLO(model_path)
    return model, face_mesh


def init_camera(camera_index, width, height):
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)  # 🔥 faster on Windows

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera {camera_index}. Check if another app is using it."
        )

    # Set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    # 🔥 Reduce internal buffering (VERY IMPORTANT)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, width, height)

    return cap

def calculate_ear(eye_points, landmarks, width, height, offset=(0, 0)):
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


# OLD CODE (BACKUP)
# This ran YOLO every few frames, but still used the full camera frame. On
# slower laptops that can create visible lag.
#
# def run_phone_detection(frame, model, app_state):
#     app_state["frame_count"] += 1
#
#     if app_state["frame_count"] % PHONE_DETECT_EVERY_N_FRAMES == 0:
#         results = model(frame, verbose=False)
#         best_phone = None
#
#         for result in results:
#             for box in result.boxes:
#                 cls = int(box.cls[0])
#                 confidence = float(box.conf[0])
#
#                 if cls == PHONE_CLASS_ID and confidence >= PHONE_CONFIDENCE:
#                     if best_phone is None or confidence > best_phone[0]:
#                         x1, y1, x2, y2 = map(int, box.xyxy[0])
#                         best_phone = (confidence, (x1, y1, x2, y2))
#
#         if best_phone is not None:
#             app_state["last_phone_seen_time"] = time.time()
#             app_state["phone_confidence"] = best_phone[0]
#             app_state["phone_box"] = best_phone[1]
#
#     if time.time() - app_state["last_phone_seen_time"] < PHONE_HOLD_TIME:
#         app_state["phone_detected"] = True
#     else:
#         app_state["phone_detected"] = False
#         app_state["phone_box"] = None
#         app_state["phone_confidence"] = 0.0


# NEW OPTIMIZED CODE
def run_phone_detection(frame, model, app_state):
    app_state["frame_count"] += 1

    # Send frame to YOLO thread occasionally
    if app_state["frame_count"] % PHONE_DETECT_EVERY_N_FRAMES == 0:
        app_state["latest_frame_for_yolo"] = frame.copy()

    # Use last YOLO result (non-blocking)
    if time.time() - app_state["last_phone_seen_time"] < PHONE_HOLD_TIME:
        app_state["phone_detected"] = True
    else:
        app_state["phone_detected"] = False
        app_state["phone_box"] = None
        app_state["phone_confidence"] = 0.0

def resize_frame_for_width(frame, target_width):
    height, width = frame.shape[:2]
    if width <= target_width:
        return frame, 1.0

    scale = width / target_width
    target_height = int(height / scale)
    resized = cv2.resize(frame, (target_width, target_height))
    return resized, scale


def scale_xyxy_box(box, scale):
    x1, y1, x2, y2 = box
    return (
        int(x1 * scale),
        int(y1 * scale),
        int(x2 * scale),
        int(y2 * scale),
    )


# OLD CODE (BACKUP)
# This processed the first face returned by MediaPipe. It worked for a single
# person, but it could accidentally process a passenger if multiple faces were
# visible in the camera frame.
#
# def run_face_detection(frame, face_mesh):
#     height, width, _ = frame.shape
#     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     result = face_mesh.process(rgb_frame)
#
#     if not result.multi_face_landmarks:
#         return None, None, None
#
#     landmarks = result.multi_face_landmarks[0].landmark
#     left_ear, left_coords = calculate_ear(LEFT_EYE, landmarks, width, height)
#     right_ear, right_coords = calculate_ear(RIGHT_EYE, landmarks, width, height)
#     ear = (left_ear + right_ear) / 2.0
#
#     return ear, left_coords, right_coords


# NEW CODE (FACE RE-IDENTIFICATION)
def get_face_recognition():
    try:
        import face_recognition
    except ImportError as exc:
        raise RuntimeError(
            "The face_recognition package is required for driver-specific "
            "tracking. Install it with: pip install face_recognition"
        ) from exc

    return face_recognition


def get_face_locations(frame, process_width=None):
    face_recognition = get_face_recognition()
    process_frame, scale = resize_frame_for_width(frame, process_width or frame.shape[1])
    rgb_frame = cv2.cvtColor(process_frame, cv2.COLOR_BGR2RGB)
    rgb_frame = np.ascontiguousarray(rgb_frame, dtype=np.uint8)
    face_locations = face_recognition.face_locations(rgb_frame)
    return rgb_frame, face_locations, scale


def get_largest_face_location(face_locations):
    if not face_locations:
        return None

    return max(face_locations, key=face_area)


def face_area(face_location):
    top, right, bottom, left = face_location
    return (right - left) * (bottom - top)


def extract_face_encoding(rgb_frame, face_location):
    face_recognition = get_face_recognition()

    # A face encoding is a 128-number fingerprint for one detected face.
    encodings = face_recognition.face_encodings(rgb_frame, [face_location])

    if not encodings:
        return None

    return encodings[0]


def compare_face_encoding(driver_encoding, face_encoding, tolerance=FACE_MATCH_TOLERANCE):
    face_recognition = get_face_recognition()

    # Lower tolerance is stricter. 0.5 is a practical starting point for webcam use.
    matches = face_recognition.compare_faces(
        [driver_encoding],
        face_encoding,
        tolerance=tolerance,
    )
    distance = face_recognition.face_distance([driver_encoding], face_encoding)[0]
    return matches[0], distance


# OLD CODE (BACKUP)
# This re-ran face recognition on the full frame every loop. It was accurate,
# but it made the live camera feed lag because face embeddings are expensive.
#
# def find_driver_face(frame, driver_encoding, tolerance=FACE_MATCH_TOLERANCE):
#     rgb_frame, face_locations = get_face_locations(frame)
#     best_match = None
#
#     for face_location in face_locations:
#         face_encoding = extract_face_encoding(rgb_frame, face_location)
#         if face_encoding is None:
#             continue
#
#         is_match, distance = compare_face_encoding(
#             driver_encoding,
#             face_encoding,
#             tolerance=tolerance,
#         )
#
#         if is_match and (best_match is None or distance < best_match[1]):
#             best_match = (face_location, distance)
#
#     if best_match is None:
#         return None, None
#
#     return best_match


# NEW OPTIMIZED CODE
def find_driver_face(frame, driver_encoding, tolerance=FACE_MATCH_TOLERANCE):
    rgb_frame, face_locations, scale = get_face_locations(
        frame,
        process_width=FACE_RECOGNITION_WIDTH,
    )
    best_match = None

    # Check every visible face and keep the closest match to the enrolled driver.
    for face_location in face_locations:
        face_encoding = extract_face_encoding(rgb_frame, face_location)
        if face_encoding is None:
            continue

        is_match, distance = compare_face_encoding(
            driver_encoding,
            face_encoding,
            tolerance=tolerance,
        )

        if is_match and (best_match is None or distance < best_match[1]):
            best_match = (scale_face_location(face_location, scale), distance)

    if best_match is None:
        return None, None

    return best_match


def should_refresh_driver_match(app_state):
    if app_state["driver_face_location"] is None:
        return True

    current_time = time.time()
    driver_recently_seen = (
        current_time - app_state["last_driver_seen_time"] < DRIVER_HOLD_TIME
    )
    frame_due = app_state["frame_count"] % DRIVER_MATCH_EVERY_N_FRAMES == 0

    return frame_due or not driver_recently_seen


def update_driver_match(frame, driver_encoding, app_state):
    if should_refresh_driver_match(app_state):
        driver_location, match_distance = find_driver_face(frame, driver_encoding)

        if driver_location is not None:
            app_state["driver_face_location"] = driver_location
            app_state["driver_match_distance"] = match_distance
            app_state["last_driver_seen_time"] = time.time()

    if time.time() - app_state["last_driver_seen_time"] < DRIVER_HOLD_TIME:
        return app_state["driver_face_location"], app_state["driver_match_distance"]

    app_state["driver_face_location"] = None
    app_state["driver_match_distance"] = None
    return None, None


def scale_face_location(face_location, scale):
    top, right, bottom, left = face_location
    return (
        int(top * scale),
        int(right * scale),
        int(bottom * scale),
        int(left * scale),
    )


def run_driver_face_detection(frame, face_mesh, driver_face_location):
    face_box = face_location_to_xyxy(driver_face_location)
    face_box = expand_xyxy_box(face_box, margin_ratio=0.25)
    left, top, right, bottom = clamp_xyxy_box(face_box, frame.shape)
    face_roi = frame[top:bottom, left:right]

    if face_roi.size == 0:
        return None, None, None, True

    roi_height, roi_width, _ = face_roi.shape
    rgb_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
    rgb_roi = np.ascontiguousarray(rgb_roi, dtype=np.uint8)

    # MediaPipe now runs only on the matched driver's face crop, not passengers.
    result = face_mesh.process(rgb_roi)

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


def is_head_forward(landmarks):
    """Return False when the nose appears low between eyes and mouth."""
    eye_y = (landmarks[33].y + landmarks[263].y) / 2.0
    nose_y = landmarks[1].y
    mouth_y = landmarks[13].y

    eye_to_mouth = mouth_y - eye_y
    if eye_to_mouth <= 0:
        return True

    # Looking down usually moves the nose visually closer to the mouth area.
    down_score = (nose_y - eye_y) / eye_to_mouth
    return down_score <= HEAD_DOWN_THRESHOLD


def clamp_face_location(face_location, frame_shape):
    frame_height, frame_width = frame_shape[:2]
    top, right, bottom, left = face_location

    top = max(0, top)
    right = min(frame_width, right)
    bottom = min(frame_height, bottom)
    left = max(0, left)

    return top, right, bottom, left


def clamp_xyxy_box(box, frame_shape):
    frame_height, frame_width = frame_shape[:2]
    x1, y1, x2, y2 = box

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(frame_width, x2)
    y2 = min(frame_height, y2)

    return x1, y1, x2, y2


def draw_driver_box(frame, face_location, distance):
    top, right, bottom, left = clamp_face_location(face_location, frame.shape)
    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 255), 2)
    cv2.putText(
        frame,
        f"DRIVER MATCH {distance:.2f}",
        (left, max(25, top - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
    )


# OLD CODE (BACKUP)
# This only checked EAR. When the driver looked down at a phone, the eyes could
# appear narrow and trigger a false drowsy alert.
#
# def handle_drowsiness(frame, ear, app_state, ear_threshold):
#     current_time = time.time()
#
#     if ear < ear_threshold:
#         if app_state["eyes_closed_since"] is None:
#             app_state["eyes_closed_since"] = current_time
#
#         closed_duration = current_time - app_state["eyes_closed_since"]
#         if closed_duration >= DROWSY_SECONDS:
#             app_state["status"] = "DROWSY"
#             cv2.putText(
#                 frame,
#                 "DROWSY! WAKE UP!",
#                 (100, 160),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 1.5,
#                 (0, 0, 255),
#                 3,
#             )
#
#             if current_time - app_state["last_drowsy_alert"] > DROWSY_COOLDOWN:
#                 play_beep(1000, 500)
#                 app_state["last_drowsy_alert"] = current_time
#         else:
#             app_state["status"] = "EYES CLOSING"
#     else:
#         app_state["eyes_closed_since"] = None
#         app_state["status"] = "FOCUSED"


# NEW OPTIMIZED CODE
def handle_drowsiness(frame, ear, app_state, ear_threshold, suppress_reason=None):
    if suppress_reason is not None:
        app_state["eyes_closed_since"] = None
        app_state["status"] = suppress_reason
        return

    current_time = time.time()

    # 🔥 STEP 3 — Dynamic threshold (baseline-based)
    if app_state.get("calibrated") and app_state.get("baseline_ear") is not None:
        dynamic_threshold = app_state["baseline_ear"] * 0.75
    else:
        dynamic_threshold = ear_threshold

    if ear < dynamic_threshold:
        if app_state["eyes_closed_since"] is None:
            app_state["eyes_closed_since"] = current_time

        closed_duration = current_time - app_state["eyes_closed_since"]

        if closed_duration >= DROWSY_SECONDS:
            app_state["status"] = "DROWSY"

            cv2.putText(
                frame,
                "DROWSY! WAKE UP!",
                (100, 160),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (0, 0, 255),
                3,
            )

            if current_time - app_state["last_drowsy_alert"] > DROWSY_COOLDOWN:
                play_beep(1000, 500)
                app_state["last_drowsy_alert"] = current_time

        else:
            app_state["status"] = "EYES CLOSING"

    else:
        app_state["eyes_closed_since"] = None
        app_state["status"] = "FOCUSED"

# def get_drowsiness_suppression_reason(app_state, driver_face_location, head_forward):
#     if not head_forward:
#         return "LOOKING DOWN"

#     if app_state["phone_detected"] and is_phone_near_face(
#         app_state["phone_box"],
#         driver_face_location,
#     ):
#         return "PHONE NEAR FACE"

#     return None
def get_drowsiness_suppression_reason(app_state, driver_face_location, head_forward, ear):
    
    # 🔥 PRIORITY: if eyes REALLY closed → DO NOT suppress
    if ear is not None and ear < EAR_THRESHOLD * 0.8:
        return None   # allow drowsiness detection

    # Now apply suppression logic
    if not head_forward:
        return "LOOKING DOWN"

    if app_state["phone_detected"] and is_phone_near_face(
        app_state["phone_box"],
        driver_face_location,
    ):
        return "PHONE NEAR FACE"

    return None

def create_state():
    return {
        "phone_detected": False,
        "last_phone_seen_time": 0.0,
        "phone_box": None,
        "phone_confidence": 0.0,
        "last_phone_alert": 0.0,
        "last_drowsy_alert": 0.0,
        "eyes_closed_since": None,
        "frame_count": 0,
        "driver_face_location": None,
        "driver_match_distance": None,
        "last_driver_seen_time": 0.0,
        "status": "STARTING",
        "baseline_ear": None,
        "baseline_samples": [],
        "calibrated": False,
        "mode": "INITIALIZING",
        # ✅ ADD THESE (threading)
        "latest_frame_for_yolo": None,
    }

def update_mode(app_state):
    status = app_state["status"]

    if status == "DROWSY":
        app_state["mode"] = "DROWSY MODE"

    elif status in ["PHONE DETECTED", "PHONE NEAR FACE", "LOOKING DOWN"]:
        app_state["mode"] = "DISTRACTION MODE"

    elif status == "FOCUSED":
        app_state["mode"] = "FOCUSED MODE"

    elif status == "DRIVER NOT FOUND":
        app_state["mode"] = "NO DRIVER"

    else:
        app_state["mode"] = "MONITORING"


def update_baseline(ear, app_state):
    if ear is None:
        return

    # Only collect when face is stable
    if not app_state["calibrated"] and app_state["status"] == "FOCUSED":
        app_state["baseline_samples"].append(ear)

        if len(app_state["baseline_samples"]) >= 30:
            app_state["baseline_ear"] = sum(app_state["baseline_samples"]) / len(app_state["baseline_samples"])
            app_state["calibrated"] = True

def yolo_worker(model, app_state):
    while True:
        if app_state["latest_frame_for_yolo"] is None:
            time.sleep(0.01)
            continue

        frame = app_state["latest_frame_for_yolo"]
        app_state["latest_frame_for_yolo"] = None

        yolo_frame, scale = resize_frame_for_width(frame, YOLO_PROCESS_WIDTH)
        results = model(yolo_frame, verbose=False)

        best_phone = None

        for result in results:
            for box in result.boxes:
                cls = int(box.cls[0])
                confidence = float(box.conf[0])

                if cls == PHONE_CLASS_ID and confidence >= PHONE_CONFIDENCE:
                    if best_phone is None or confidence > best_phone[0]:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        phone_box = scale_xyxy_box((x1, y1, x2, y2), scale)
                        best_phone = (confidence, phone_box)

        if best_phone is not None:
            app_state["last_phone_seen_time"] = time.time()
            app_state["phone_confidence"] = best_phone[0]
            app_state["phone_box"] = best_phone[1]
            
def is_phone_near_face(phone_box, face_location):
    if phone_box is None or face_location is None:
        return False

    expanded_face_box = expand_xyxy_box(
        face_location_to_xyxy(face_location),
        PHONE_NEAR_FACE_MARGIN,
    )
    return boxes_intersect(phone_box, expanded_face_box)


def face_location_to_xyxy(face_location):
    top, right, bottom, left = face_location
    return left, top, right, bottom


def expand_xyxy_box(box, margin_ratio):
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


def boxes_intersect(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    return ax1 < bx2 and ax2 > bx1 and ay1 < by2 and ay2 > by1


def handle_phone_alert(frame, app_state):
    if not app_state["phone_detected"]:
        return

    app_state["status"] = "PHONE DETECTED"
    cv2.putText(
        frame,
        "PUT PHONE DOWN!",
        (100, 220),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (255, 0, 0),
        3,
    )

    current_time = time.time()
    if current_time - app_state["last_phone_alert"] > PHONE_COOLDOWN:
        play_beep(700, 300)
        app_state["last_phone_alert"] = current_time

    if app_state["phone_box"] is not None:
        x1, y1, x2, y2 = app_state["phone_box"]
        confidence = app_state["phone_confidence"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(
            frame,
            f"PHONE {confidence:.2f}",
            (x1, max(25, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2,
        )


def draw_hud(frame, ear, left_coords, right_coords, app_state):
    # Draw eye landmarks
    if left_coords and right_coords:
        for x, y in left_coords + right_coords:
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

    # EAR display
    ear_text = f"EAR: {ear:.2f}" if ear is not None else "EAR: --"
    cv2.putText(
        frame,
        ear_text,
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
    )

    # 🔥 NEW — MODE (high-level)
    cv2.putText(
        frame,
        f"MODE: {app_state.get('mode', '---')}",
        (30, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 0),
        2,
    )

    # STATUS (low-level)
    cv2.putText(
        frame,
        f"STATUS: {app_state['status']}",
        (30, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        status_color(app_state["status"]),
        2,
    )

    # Quit instruction
    cv2.putText(
        frame,
        "Press Q or Esc to quit",
        (30, frame.shape[0] - 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )

    # Calibration progress
    if not app_state["calibrated"]:
        samples = len(app_state["baseline_samples"])
        cv2.putText(
            frame,
            f"Calibrating ({samples}/30)... Look straight",
            (30, 140),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )

    # Baseline display
    if app_state["calibrated"]:
        cv2.putText(
            frame,
            f"Baseline EAR: {app_state['baseline_ear']:.2f}",
            (30, 170),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2,
        )

def status_color(status):
    if status == "DROWSY":
        return (0, 0, 255)
    if status == "PHONE DETECTED":
        return (255, 0, 0)
    if status == "PHONE NEAR FACE":
        return (255, 0, 0)
    if status == "LOOKING DOWN":
        return (0, 255, 255)
    if status == "DRIVER NOT FOUND":
        return (0, 255, 255)
    if status == "NO FACE":
        return (0, 255, 255)
    return (0, 255, 0)
