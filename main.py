import argparse

import cv2
import threading
from utils import *
from config import Driver_threshold
# from utils import update_baseline, update_mode, yolo_worker

# from utils import (
#     EAR_THRESHOLD,
#     WINDOW_NAME,
#     create_state,
#     draw_driver_box,
#     draw_hud,
#     extract_face_encoding,
#     get_face_locations,
#     get_largest_face_location,
#     get_drowsiness_suppression_reason,
#     handle_drowsiness,
#     handle_phone_alert,
#     init_camera,
#     init_models,
#     run_driver_face_detection,
#     run_phone_detection,
#     update_driver_match,
# )


def parse_args():
    parser = argparse.ArgumentParser(description="Drowsiness and phone detector")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--width", type=int, default=640, help="Camera/window width")
    parser.add_argument("--height", type=int, default=480, help="Camera/window height")
    parser.add_argument("--model", default="yolov8n.pt", help="YOLO model path")
    parser.add_argument(
        "--ear-threshold",
        type=float,
        default=EAR_THRESHOLD,
        help="Eye aspect ratio threshold for closed eyes",
    )
    return parser.parse_args()


# OLD CODE (BACKUP)
# This version processed whichever face MediaPipe returned first. It is kept
# here as a backup, but the working code below uses face re-identification so
# only the enrolled driver is checked for drowsiness.
#
# def main():
#     args = parse_args()
#     app_state = create_state()
#     model, face_mesh = init_models(args.model)
#     cap = init_camera(args.camera, args.width, args.height)
#
#     try:
#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 app_state["status"] = "CAMERA ERROR"
#                 break
#
#             run_phone_detection(frame, model, app_state)
#             ear, left_coords, right_coords = run_face_detection(frame, face_mesh)
#
#             if ear is not None:
#                 handle_drowsiness(frame, ear, app_state, args.ear_threshold)
#             else:
#                 app_state["eyes_closed_since"] = None
#                 app_state["status"] = "NO FACE"
#
#             handle_phone_alert(frame, app_state)
#             draw_hud(frame, ear, left_coords, right_coords, app_state)
#
#             cv2.imshow(WINDOW_NAME, frame)
#             if cv2.waitKey(1) & 0xFF in (27, ord("q")):
#                 break
#     finally:
#         cap.release()
#         face_mesh.close()
#         cv2.destroyAllWindows()


# NEW CODE (FACE RE-IDENTIFICATION)
def capture_driver_encoding(cap):
    """Enroll the driver by saving one face embedding before monitoring starts."""
    while True:
        ret, frame = cap.read()
        if not ret:
            raise RuntimeError("Could not read from camera while enrolling driver.")

        # Detect visible faces and pick the largest one as the driver during setup.
        rgb_frame, face_locations, _ = get_face_locations(frame)
        driver_location = get_largest_face_location(face_locations)

        cv2.putText(
            frame,
            "Driver setup: look at camera, then press C",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            Driver_threshold,
            (0, 255, 255),
            2,
        )
        cv2.putText(
            frame,
            "Press Q or Esc to quit",
            (30, frame.shape[0] - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        if driver_location:
            top, right, bottom, left = driver_location
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 255), 2)
            cv2.putText(
                frame,
                "Face found",
                (left, max(25, top - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )
        else:
            cv2.putText(
                frame,
                "No face found yet",
                (30, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
            )

        cv2.imshow(WINDOW_NAME, frame)
        key = cv2.waitKey(1) & 0xFF

        if key in (27, ord("q")):
            return None

        if key == ord("c") and driver_location is not None:
            # Convert the selected driver face into a numeric embedding.
            driver_encoding = extract_face_encoding(rgb_frame, driver_location)
            if driver_encoding is not None:
                return driver_encoding


# OLD CODE (BACKUP)
# This face re-identification loop recomputed driver embeddings every frame.
# It worked, but it was too slow for real-time video on many machines.
#
# def main():
#     args = parse_args()
#     app_state = create_state()
#     model, face_mesh = init_models(args.model)
#     cap = init_camera(args.camera, args.width, args.height)
#
#     try:
#         driver_encoding = capture_driver_encoding(cap)
#         if driver_encoding is None:
#             return
#
#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 app_state["status"] = "CAMERA ERROR"
#                 break
#
#             run_phone_detection(frame, model, app_state)
#             driver_location, match_distance = find_driver_face(
#                 frame,
#                 driver_encoding,
#                 tolerance=FACE_MATCH_TOLERANCE,
#             )
#
#             if driver_location is not None:
#                 draw_driver_box(frame, driver_location, match_distance)
#                 ear, left_coords, right_coords = run_driver_face_detection(
#                     frame,
#                     face_mesh,
#                     driver_location,
#                 )
#             else:
#                 ear, left_coords, right_coords = None, None, None
#
#             if driver_location is not None and ear is not None:
#                 handle_drowsiness(frame, ear, app_state, args.ear_threshold)
#             else:
#                 app_state["eyes_closed_since"] = None
#                 app_state["status"] = "DRIVER NOT FOUND"
#
#             handle_phone_alert(frame, app_state)
#             draw_hud(frame, ear, left_coords, right_coords, app_state)
#
#             cv2.imshow(WINDOW_NAME, frame)
#             if cv2.waitKey(1) & 0xFF in (27, ord("q")):
#                 break
#     finally:
#         cap.release()
#         face_mesh.close()
#         cv2.destroyAllWindows()


# NEW OPTIMIZED CODE

def main():
    args = parse_args()
    app_state = create_state()
    model, face_mesh = init_models(args.model)
    cap = init_camera(args.camera, args.width, args.height)
    threading.Thread(target=yolo_worker, args=(model, app_state), daemon=True).start()
    try:
        # Step 1: Capture and store the driver's face encoding.
        driver_encoding = capture_driver_encoding(cap)
        if driver_encoding is None:
            return
    
        while True:
            ret, frame = cap.read()
            if not ret:
                app_state["status"] = "CAMERA ERROR"
                break

            run_phone_detection(frame, model, app_state)

            # Step 2: Re-identify the driver only every few frames, then reuse
            # the recent match briefly. This removes most embedding work.
            driver_location, match_distance = update_driver_match(
                frame,
                driver_encoding,
                app_state,
            )

            # Step 3: Process drowsiness only for the matched driver face.
            if driver_location is not None:
                draw_driver_box(frame, driver_location, match_distance)
                ear, left_coords, right_coords, head_forward = run_driver_face_detection(
                    frame,
                    face_mesh,
                    driver_location,
                )
            else:
                ear, left_coords, right_coords = None, None, None
                head_forward = True

            update_baseline(ear, app_state)
            if driver_location is not None and ear is not None:
                suppress_reason = get_drowsiness_suppression_reason(
                    app_state,
                    driver_location,
                    head_forward,
                    ear
                )
                handle_drowsiness(
                    frame,
                    ear,
                    app_state,
                    args.ear_threshold,
                    suppress_reason=suppress_reason,
                )
            else:
                app_state["eyes_closed_since"] = None 
                app_state["status"] = "DRIVER NOT FOUND"

            handle_phone_alert(frame, app_state)
            update_mode(app_state)
            draw_hud(frame, ear, left_coords, right_coords, app_state)

            cv2.imshow(WINDOW_NAME, frame)
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                break
    finally:
        cap.release()
        face_mesh.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
