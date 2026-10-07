import json
import os
import time
from concurrent.futures import Future, ThreadPoolExecutor
from collections import deque
from dataclasses import dataclass
from math import hypot
from pathlib import Path
from statistics import median
from typing import Optional

import cv2
import torch
from ultralytics import YOLO


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CAMERA_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
PROJECT_DIRECTORY = Path(__file__).resolve().parents[1]
POSE_MODEL_PATH = PROJECT_DIRECTORY / "yolov8n-pose.pt"
FALLBACK_MODEL_PATH = PROJECT_DIRECTORY / "yolov8n.pt"
INFERENCE_SIZE = 256
CPU_THREADS = 2
TERMINAL_REPORT_INTERVAL_SECONDS = 1.0
MODEL_UPDATE_INTERVAL = 2
QR_SCAN_INTERVAL = 5
USE_CUDA = torch.cuda.is_available()
FEET_PER_METER = 3.28084
REAL_PERSON_HEIGHT = 1.76784
FOCAL_LENGTH_PIXELS: Optional[float] = None
STOP_DISTANCE = 1.0
WARNING_DISTANCE = 2.0
CONFIDENCE_THRESHOLD = 0.35
CALIBRATION_FILE = PROJECT_DIRECTORY / "camera_calibration.json"
PERSON_CLASS_ID = 0
WINDOW_NAME = "Robot Human Distance"
KEYPOINT_NAMES = (
    "Nose",
    "L-eye",
    "R-eye",
    "L-ear",
    "R-ear",
    "L-shoulder",
    "R-shoulder",
    "L-elbow",
    "R-elbow",
    "L-wrist",
    "R-wrist",
    "L-hip",
    "R-hip",
    "L-knee",
    "R-knee",
    "L-ankle",
    "R-ankle",
)
SKELETON_LINKS = (
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
)
PERSON_COLORS = [
    (255, 180, 50),
    (60, 220, 120),
    (220, 120, 220),
    (80, 200, 240),
    (220, 180, 70),
]
QR_RETRY_INTERVAL = 3
QR_LOST_FRAME_LIMIT = 3
QR_CONFIRMATION_WINDOW = 3


@dataclass
class PersonDetection:
    person_id: str
    bbox: tuple[int, int, int, int]
    center: tuple[int, int]
    width: int
    height: int
    confidence: float
    distance: Optional[float]
    horizontal_position: str
    vertical_position: str
    partial: bool = False
    distance_from_partial_pose: bool = False
    keypoints: tuple[tuple[float, float, float], ...] = ()


# ---------------------------------------------------------------------------
# Camera and source handling
# ---------------------------------------------------------------------------

def initialize_camera() -> cv2.VideoCapture:
    """Open either a local webcam or a file-based video source."""
    source = (
        os.environ.get("MOTOR_VIDEO_SOURCE")
        or os.environ.get("VIDEO_SOURCE")
        or (os.sys.argv[1] if len(os.sys.argv) > 1 else None)
    )

    if source:
        if str(source).isdigit():
            camera = cv2.VideoCapture(int(source))
            if camera.isOpened():
                camera.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
                camera.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
                camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                print(f"Using camera index {source}.")
                return camera
            camera.release()

        file_path = Path(str(source)).expanduser()
        if file_path.exists() and file_path.is_file():
            camera = cv2.VideoCapture(str(file_path))
            if camera.isOpened():
                print(f"Using video source: {file_path}")
                return camera
            camera.release()

        raise RuntimeError(
            f"Could not open the requested source: {source}. "
            "Set MOTOR_VIDEO_SOURCE to a valid camera index or video path."
        )

    camera = cv2.VideoCapture(CAMERA_INDEX)
    if not camera.isOpened():
        camera.release()
        raise RuntimeError(
            f"Could not open camera {CAMERA_INDEX}. Check the connection and "
            "change CAMERA_INDEX near the top of this file if needed."
        )

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return camera


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model() -> YOLO:
    """Load the preferred person-detection model, falling back if needed."""
    if not USE_CUDA:
        torch.set_num_threads(CPU_THREADS)

    if POSE_MODEL_PATH.is_file():
        return YOLO(str(POSE_MODEL_PATH))

    if FALLBACK_MODEL_PATH.is_file():
        print(
            f"{POSE_MODEL_PATH} is not installed. Using {FALLBACK_MODEL_PATH}; "
            "partial-body distances need pose weights and will remain unknown."
        )
        return YOLO(str(FALLBACK_MODEL_PATH))

    try:
        return YOLO(str(POSE_MODEL_PATH))
    except Exception:
        return YOLO(str(FALLBACK_MODEL_PATH))


# ---------------------------------------------------------------------------
# Distance / pose estimation
# ---------------------------------------------------------------------------

def format_distance(distance: float) -> str:
    return f"{distance:.2f} m / {distance * FEET_PER_METER:.2f} ft"


def load_calibration() -> Optional[float]:
    if not CALIBRATION_FILE.is_file():
        return FOCAL_LENGTH_PIXELS

    try:
        with CALIBRATION_FILE.open("r", encoding="utf-8") as file:
            value = float(json.load(file)["focal_length_pixels"])
        return value if value > 0 else FOCAL_LENGTH_PIXELS
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        print(f"Ignoring invalid calibration file: {CALIBRATION_FILE}")
        return FOCAL_LENGTH_PIXELS


def calculate_distance(
    pixel_height: float,
    real_height: float = REAL_PERSON_HEIGHT,
    focal_length: Optional[float] = FOCAL_LENGTH_PIXELS,
) -> Optional[float]:
    if pixel_height <= 0 or real_height <= 0 or focal_length is None or focal_length <= 0:
        return None
    return (real_height * focal_length) / pixel_height


def estimate_full_person_pixel_height(
    keypoints: Optional[list[list[float]]],
    visible_height: int,
) -> Optional[float]:
    """Estimate full person height from visible pose segments."""
    if not keypoints:
        return None

    def joint(index: int) -> Optional[tuple[float, float]]:
        if index >= len(keypoints) or len(keypoints[index]) < 3:
            return None
        x, y, confidence = keypoints[index][:3]
        if float(confidence) < 0.35:
            return None
        return float(x), float(y)

    def joint_midpoint(indices: tuple[int, int]) -> Optional[tuple[float, float]]:
        points = []
        for idx in indices:
            point = joint(idx)
            if point is not None:
                points.append(point)
        if not points:
            return None
        return (
            sum(point[0] for point in points) / len(points),
            sum(point[1] for point in points) / len(points),
        )

    head = joint(0)
    shoulders = joint_midpoint((5, 6))
    hips = joint_midpoint((11, 12))
    knees = joint_midpoint((13, 14))
    ankles = joint_midpoint((15, 16))

    segments = [
        (head, shoulders, 0.12),
        (shoulders, hips, 0.29),
        (hips, knees, 0.245),
        (knees, ankles, 0.246),
        (hips, ankles, 0.49),
        (shoulders, knees, 0.535),
        (head, ankles, 0.92),
    ]
    estimates = []
    for start, end, fraction in segments:
        if start is None or end is None:
            continue
        segment_pixels = hypot(end[0] - start[0], end[1] - start[1])
        full_height = segment_pixels / fraction
        if full_height >= visible_height * 0.95:
            estimates.append(full_height)

    return median(estimates) if estimates else None


def calculate_position(center: tuple[int, int], frame_width: int, frame_height: int) -> tuple[str, str]:
    x, y = center
    horizontal = "LEFT" if x < frame_width / 3 else "RIGHT" if x > frame_width * 2 / 3 else "CENTER"
    vertical = "TOP" if y < frame_height / 3 else "BOTTOM" if y > frame_height * 2 / 3 else "CENTER"
    return horizontal, vertical


# ---------------------------------------------------------------------------
# Person detection
# ---------------------------------------------------------------------------

def detect_people(
    model: YOLO,
    frame,
    focal_length: Optional[float],
    track_labels: dict[int, str],
    next_label: int,
) -> tuple[list[PersonDetection], int]:
    """Detect person boxes and estimate distance using bounding-box height."""
    if not USE_CUDA:
        torch.set_num_threads(CPU_THREADS)

    results = model.track(
        frame,
        persist=True,
        classes=[PERSON_CLASS_ID],
        conf=CONFIDENCE_THRESHOLD,
        imgsz=INFERENCE_SIZE,
        device=0 if USE_CUDA else "cpu",
        max_det=20,
        verbose=False,
    )
    if not results or results[0].boxes is None:
        return [], next_label

    frame_height, frame_width = frame.shape[:2]
    detections: list[PersonDetection] = []
    pose_keypoints = getattr(results[0], "keypoints", None)

    for box_index, box in enumerate(results[0].boxes):
        confidence = float(box.conf[0].item())
        if confidence < CONFIDENCE_THRESHOLD or int(box.cls[0].item()) != PERSON_CLASS_ID:
            continue

        x1, y1, x2, y2 = (int(round(value)) for value in box.xyxy[0].tolist())
        x1 = max(0, min(x1, frame_width - 1))
        y1 = max(0, min(y1, frame_height - 1))
        x2 = max(0, min(x2, frame_width - 1))
        y2 = max(0, min(y2, frame_height - 1))
        width = x2 - x1
        height = y2 - y1
        if width <= 0 or height <= 0:
            continue

        keypoint_data = None
        if pose_keypoints is not None and pose_keypoints.data is not None:
            keypoint_data = pose_keypoints.data[box_index].detach().cpu().tolist()

        pose_partial = False
        if keypoint_data and len(keypoint_data) >= 17:
            head_visible = keypoint_data[0][2] >= 0.35
            ankle_visible = keypoint_data[15][2] >= 0.35 or keypoint_data[16][2] >= 0.35
            pose_partial = not head_visible or not ankle_visible

        touches_edge = x1 <= 1 or y1 <= 1 or x2 >= frame_width - 2 or y2 >= frame_height - 2
        partial = touches_edge or pose_partial

        pose_height = estimate_full_person_pixel_height(keypoint_data, height) if keypoint_data else None
        person_pixel_height = pose_height if pose_height is not None else height
        person_keypoints = tuple(tuple(map(float, point[:3])) for point in keypoint_data or ())

        tracker_id = int(box.id[0].item()) if box.id is not None else None
        if tracker_id is not None:
            if tracker_id not in track_labels:
                track_labels[tracker_id] = f"Person {next_label}"
                next_label += 1
            person_id = track_labels[tracker_id]
        else:
            person_id = f"Person {next_label}"
            next_label += 1

        center = ((x1 + x2) // 2, (y1 + y2) // 2)
        horizontal, vertical = calculate_position(center, frame_width, frame_height)
        measured_distance = (
            calculate_distance(person_pixel_height, focal_length=focal_length)
            if (not partial or pose_height is not None)
            else None
        )

        detections.append(
            PersonDetection(
                person_id=person_id,
                bbox=(x1, y1, x2, y2),
                center=center,
                width=width,
                height=height,
                confidence=confidence,
                distance=measured_distance,
                horizontal_position=horizontal,
                vertical_position=vertical,
                partial=partial,
                distance_from_partial_pose=partial and pose_height is not None,
                keypoints=person_keypoints,
            )
        )

    return detections, next_label


def find_nearest_person(people: list[PersonDetection]) -> Optional[PersonDetection]:
    valid = [person for person in people if person.distance is not None]
    if not valid:
        return None
    return min(valid, key=lambda person: person.distance or float("inf"))


def get_robot_status(people: list[PersonDetection], nearest: Optional[PersonDetection]) -> str:
    if not people:
        return "SAFE"
    nearest_distance = nearest.distance if nearest is not None else None
    if nearest_distance is not None and nearest_distance <= STOP_DISTANCE:
        return "STOP"
    if nearest_distance is not None and nearest_distance <= WARNING_DISTANCE:
        return "WARNING"
    if nearest_distance is None:
        return "PERSON_DETECTED"
    return "SAFE"


# ---------------------------------------------------------------------------
# QR / ArUco scanning
# ---------------------------------------------------------------------------

def initialize_qr_detectors():
    aruco = cv2.aruco
    parameters = aruco.DetectorParameters()
    parameters.minMarkerPerimeterRate = 0.02
    parameters.adaptiveThreshWinSizeMax = 33
    parameters.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
    marker_detector = aruco.ArucoDetector(
        aruco.getPredefinedDictionary(aruco.DICT_4X4_50),
        parameters,
    )
    qr_detector = cv2.QRCodeDetector()
    contrast_enhancer = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return qr_detector, marker_detector, contrast_enhancer


def scan_qr_and_markers(frame, frame_count, qr_detector, marker_detector, contrast_enhancer):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = marker_detector.detectMarkers(gray)
    qr_data = None
    detection_type = "ARUCO MARKER"
    boundary = None

    if ids is not None:
        for marker_index, marker_id in enumerate(ids.flatten()):
            if 1 <= int(marker_id) <= 10:
                qr_data = str(int(marker_id))
                boundary = corners[marker_index].reshape(-1, 2)
                break

    if qr_data is None and frame_count % QR_RETRY_INTERVAL == 0:
        enhanced = contrast_enhancer.apply(gray)
        enlarged = cv2.resize(enhanced, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
        retry_corners, retry_ids, _ = marker_detector.detectMarkers(enlarged)
        if retry_ids is not None:
            for marker_index, marker_id in enumerate(retry_ids.flatten()):
                if 1 <= int(marker_id) <= 10:
                    qr_data = str(int(marker_id))
                    boundary = retry_corners[marker_index].reshape(-1, 2) / 1.5
                    break

    if qr_data is None:
        qr_text, qr_points, _ = qr_detector.detectAndDecode(gray)
        if not qr_text:
            if qr_points is not None:
                points = qr_points.reshape(-1, 2)
                x_min, y_min = points.min(axis=0).astype(int)
                x_max, y_max = points.max(axis=0).astype(int)
                margin = max(4, int(max(x_max - x_min, y_max - y_min) * 0.12))
                crop_x1 = max(0, x_min - margin)
                crop_y1 = max(0, y_min - margin)
                crop_x2 = min(gray.shape[1], x_max + margin + 1)
                crop_y2 = min(gray.shape[0], y_max + margin + 1)
                qr_crop = gray[crop_y1:crop_y2, crop_x1:crop_x2]
                scale = max(2.0, min(4.0, 160.0 / max(qr_crop.shape)))
                retry_image = cv2.resize(
                    qr_crop,
                    None,
                    fx=scale,
                    fy=scale,
                    interpolation=cv2.INTER_CUBIC,
                )
                qr_text, retry_points, _ = qr_detector.detectAndDecode(retry_image)
                if retry_points is not None:
                    qr_points = retry_points / scale
                    qr_points[..., 0] += crop_x1
                    qr_points[..., 1] += crop_y1
            else:
                scale = 1.5
                retry_image = cv2.resize(
                    gray,
                    None,
                    fx=scale,
                    fy=scale,
                    interpolation=cv2.INTER_CUBIC,
                )
                qr_text, retry_points, _ = qr_detector.detectAndDecode(retry_image)
                if retry_points is not None:
                    qr_points = retry_points / scale

        if qr_points is not None:
            boundary = qr_points.reshape(-1, 2)
        if isinstance(qr_text, str) and qr_text.strip():
            qr_data = qr_text.strip()
            detection_type = "QR CODE"

    return qr_data, detection_type, boundary if qr_data is not None else None


def update_qr_confirmation(qr_data, detection_type, recent_values, frames_without_marker, last_qr_type, last_qr_data):
    current_value = (detection_type, qr_data) if qr_data is not None else None
    recent_values.append(current_value)

    if qr_data is not None:
        frames_without_marker = 0
        if recent_values.count(current_value) >= 1:
            last_qr_type = detection_type
            last_qr_data = qr_data
    else:
        frames_without_marker += 1
        if frames_without_marker >= QR_LOST_FRAME_LIMIT:
            last_qr_type = None
            last_qr_data = None
            recent_values.clear()

    return frames_without_marker, last_qr_type, last_qr_data


# ---------------------------------------------------------------------------
# Drawing utilities
# ---------------------------------------------------------------------------

def draw_camera_reference(frame):
    height, width = frame.shape[:2]
    camera_point = (width // 2, height - 24)
    cv2.line(frame, camera_point, (int(width * 0.08), 0), (95, 95, 95), 1)
    cv2.line(frame, camera_point, (int(width * 0.92), 0), (95, 95, 95), 1)
    cv2.circle(frame, camera_point, 7, (255, 255, 255), -1)
    cv2.circle(frame, camera_point, 10, (25, 25, 25), 2)
    cv2.putText(
        frame,
        "CAMERA",
        (camera_point[0] + 13, camera_point[1] + 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return camera_point


def draw_distance_arrow(frame, camera_point, person, color):
    cv2.arrowedLine(frame, camera_point, person.center, color, 2, cv2.LINE_AA, tipLength=0.04)
    if person.distance is None:
        label = f"{person.person_id}: calibrate" if not person.partial else f"{person.person_id}: partial"
    elif person.distance_from_partial_pose:
        label = f"{person.person_id}: ~{format_distance(person.distance)}"
    else:
        label = f"{person.person_id}: {format_distance(person.distance)}"

    text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2)
    text_x = max(4, min(frame.shape[1] - text_size[0] - 8, person.center[0] + 10))
    text_y = max(text_size[1] + 8, person.center[1] - 12)
    cv2.rectangle(frame, (text_x - 4, text_y - text_size[1] - 6), (text_x + text_size[0] + 4, text_y + 4), (20, 20, 20), -1)
    cv2.putText(frame, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2, cv2.LINE_AA)


def draw_person_box(frame, person, is_nearest, color):
    x1, y1, x2, y2 = person.bbox
    box_color = (0, 128, 255) if person.partial else (0, 0, 255) if is_nearest else color
    thickness = 4 if is_nearest or person.partial else 2
    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, thickness)

    if person.distance is None:
        distance_text = "partial: distance unknown" if person.partial else "uncalibrated"
    else:
        prefix = "~" if person.distance_from_partial_pose else ""
        distance_text = f"{prefix}{format_distance(person.distance)}"

    label_lines = [
        f"{person.person_id}  {distance_text}",
        f"{person.horizontal_position}/{person.vertical_position}  {person.confidence:.0%}",
    ]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.58
    line_height = 22
    label_height = len(label_lines) * line_height + 8
    label_width = max(cv2.getTextSize(line, font, font_scale, 2)[0][0] for line in label_lines) + 12
    label_x = max(0, min(x1, frame.shape[1] - label_width))
    label_y = max(label_height, y1)
    cv2.rectangle(frame, (label_x, label_y - label_height), (label_x + label_width, label_y), box_color, -1)
    for index, line in enumerate(label_lines):
        cv2.putText(frame, line, (label_x + 6, label_y - label_height + 20 + index * line_height), font, font_scale, (255, 255, 255), 2, cv2.LINE_AA)


def draw_person_keypoints(frame, person):
    if not person.keypoints:
        return
    visible = [point[2] >= 0.35 for point in person.keypoints[: len(KEYPOINT_NAMES)]]
    for start_index, end_index in SKELETON_LINKS:
        if start_index >= len(visible) or end_index >= len(visible):
            continue
        if not visible[start_index] or not visible[end_index]:
            continue
        point_a = tuple(int(round(value)) for value in person.keypoints[start_index][:2])
        point_b = tuple(int(round(value)) for value in person.keypoints[end_index][:2])
        cv2.line(frame, point_a, point_b, (40, 230, 230), 2, cv2.LINE_AA)

    for index, (x, y, confidence) in enumerate(person.keypoints[: len(KEYPOINT_NAMES)]):
        if confidence < 0.35:
            continue
        point = (int(round(x)), int(round(y)))
        cv2.circle(frame, point, 4, (0, 40, 255), -1, cv2.LINE_AA)
        label_x = max(2, min(point[0] + 5, frame.shape[1] - 58))
        label_y = max(12, min(point[1] - 5, frame.shape[0] - 4))
        cv2.putText(frame, KEYPOINT_NAMES[index], (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 255), 1, cv2.LINE_AA)


def draw_qr_detection(frame, boundary, detection_type, qr_data):
    if boundary is None or qr_data is None:
        return

    polygon = boundary.astype(int).reshape(-1, 1, 2)
    cv2.polylines(frame, [polygon], True, (0, 255, 0), 3, cv2.LINE_AA)

    points = boundary.reshape(-1, 2)
    x_min, y_min = points.min(axis=0).astype(int)
    label = f"{detection_type}: {qr_data}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.65
    text_size, _ = cv2.getTextSize(label, font, font_scale, 2)
    text_x = max(4, min(int(x_min), frame.shape[1] - text_size[0] - 12))
    text_y = max(text_size[1] + 10, int(y_min) - 8)
    cv2.rectangle(
        frame,
        (text_x - 4, text_y - text_size[1] - 6),
        (text_x + text_size[0] + 4, text_y + 4),
        (0, 120, 0),
        -1,
    )
    cv2.putText(frame, label, (text_x, text_y), font, font_scale, (255, 255, 255), 2, cv2.LINE_AA)


def draw_status(frame, people, nearest, status, fps, focal_length):
    if nearest is None:
        nearest_text = "NEAREST PERSON: UNKNOWN"
        distance_text = "DISTANCE: N/A"
    else:
        nearest_text = f"NEAREST PERSON: {nearest.person_id}"
        distance_text = f"DISTANCE: {format_distance(nearest.distance)}" if nearest.distance is not None else "DISTANCE: N/A"

    focal_text = f"FOCAL: {focal_length:.1f} px" if focal_length else "FOCAL: not calibrated"
    lines = [
        nearest_text,
        distance_text,
        f"STATUS: {status}",
        f"PEOPLE: {len(people)}   FPS: {fps:.1f}",
        focal_text,
    ]
    panel_width = 340
    panel_height = len(lines) * 24 + 12
    cv2.rectangle(frame, (10, 10), (10 + panel_width, 10 + panel_height), (20, 20, 20), -1)
    status_color = (50, 220, 90) if status == "SAFE" else (0, 165, 255)
    if status == "STOP":
        status_color = (0, 0, 255)
    for index, line in enumerate(lines):
        color = status_color if line.startswith("STATUS:") else (245, 245, 245)
        cv2.putText(frame, line, (20, 34 + index * 24), cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 2, cv2.LINE_AA)

    if not people:
        cv2.putText(frame, "No person detected", (20, 10 + panel_height + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (235, 235, 235), 2, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    global FOCAL_LENGTH_PIXELS

    try:
        camera = initialize_camera()
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc

    try:
        model = load_model()
    except Exception as exc:
        camera.release()
        raise SystemExit(f"Could not load model: {exc}") from exc

    qr_detector, marker_detector, contrast_enhancer = initialize_qr_detectors()
    FOCAL_LENGTH_PIXELS = load_calibration()
    track_labels: dict[int, str] = {}
    next_label = 1
    distance_history: dict[str, list[float]] = {}
    inference_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="vision")
    inference_future: Optional[Future] = None
    qr_future: Optional[Future] = None
    people: list[PersonDetection] = []
    recent_qr_values = deque(maxlen=QR_CONFIRMATION_WINDOW)
    frames_without_marker = 0
    last_qr_type = None
    last_qr_data = None
    last_qr_boundary = None
    last_reported_state = None
    fps = 0.0
    previous_frame_time = time.perf_counter()
    last_terminal_report = previous_frame_time
    frame_count = 0

    device_name = "CUDA" if USE_CUDA else "CPU"
    print(f"[START] Camera {FRAME_WIDTH}x{FRAME_HEIGHT} | model device: {device_name} | press Q to quit")

    try:
        while True:
            ok, frame = camera.read()
            if not ok or frame is None:
                print("Camera frame unavailable; stopping safely.")
                break

            frame_count += 1
            current_time = time.perf_counter()
            delta = current_time - previous_frame_time
            previous_frame_time = current_time
            if delta > 1e-6:
                fps = (fps * 0.85) + ((1.0 / delta) * 0.15)

            if inference_future is not None and inference_future.done():
                try:
                    people, next_label = inference_future.result()
                except Exception as exc:
                    print(f"[WARN] Person detection failed: {exc}", flush=True)
                inference_future = None

            if qr_future is not None and qr_future.done():
                try:
                    qr_data, detection_type, boundary = qr_future.result()
                    previous_qr = (last_qr_type, last_qr_data)
                    frames_without_marker, last_qr_type, last_qr_data = update_qr_confirmation(
                        qr_data,
                        detection_type,
                        recent_qr_values,
                        frames_without_marker,
                        last_qr_type,
                        last_qr_data,
                    )
                    if qr_data is not None:
                        last_qr_boundary = boundary
                        current_qr = (detection_type, qr_data)
                        if current_qr != previous_qr:
                            print(f"[QR] {detection_type}: {qr_data}", flush=True)
                    elif last_qr_data is None:
                        last_qr_boundary = None
                except Exception as exc:
                    print(f"[WARN] QR/ArUco scan failed: {exc}", flush=True)
                qr_future = None

            if inference_future is None and frame_count % MODEL_UPDATE_INTERVAL == 1:
                inference_future = inference_executor.submit(
                    detect_people,
                    model,
                    frame.copy(),
                    FOCAL_LENGTH_PIXELS,
                    track_labels,
                    next_label,
                )

            if qr_future is None and frame_count % QR_SCAN_INTERVAL == 1:
                qr_future = inference_executor.submit(
                    scan_qr_and_markers,
                    frame.copy(),
                    frame_count,
                    qr_detector,
                    marker_detector,
                    contrast_enhancer,
                )

            if people:
                nearest = find_nearest_person(people)
                nearest_distance = nearest.distance if nearest is not None else None
                status = get_robot_status(people, nearest)
            else:
                nearest = None
                nearest_distance = None
                status = "SAFE"

            camera_point = draw_camera_reference(frame)
            for index, person in enumerate(people):
                color = PERSON_COLORS[index % len(PERSON_COLORS)]
                draw_distance_arrow(frame, camera_point, person, color)
                draw_person_box(frame, person, person is nearest, color)
                if person.keypoints:
                    draw_person_keypoints(frame, person)

            draw_status(frame, people, nearest, status, fps, FOCAL_LENGTH_PIXELS)
            draw_qr_detection(frame, last_qr_boundary, last_qr_type, last_qr_data)

            if current_time - last_terminal_report >= TERMINAL_REPORT_INTERVAL_SECONDS:
                nearest_name = nearest.person_id if nearest is not None else "none"
                distance_text = (
                    format_distance(nearest.distance)
                    if nearest is not None and nearest.distance is not None
                    else "unknown"
                )
                qr_text = f"{last_qr_type} {last_qr_data}" if last_qr_data is not None else "none"
                print(
                    f"[{time.strftime('%H:%M:%S')}] FPS={fps:.1f} | people={len(people)} "
                    f"| nearest={nearest_name} ({distance_text}) | status={status} | QR={qr_text}",
                    flush=True,
                )
                last_terminal_report = current_time

            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break

    finally:
        inference_executor.shutdown(wait=True, cancel_futures=True)
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
