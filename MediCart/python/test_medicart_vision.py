import unittest
from collections import deque
from contextlib import redirect_stdout
from io import StringIO

import cv2
import numpy as np

from medicart_vision import (
    FEET_PER_METER,
    HEARTBEAT_INTERVAL_SECONDS,
    QR_CONFIRMATION_WINDOW,
    arduino_safety_command,
    calculate_distance,
    find_nearest_person,
    get_robot_status,
    heartbeat_due,
    initialize_qr_detectors,
    log_frame_state,
    pending_qr_command,
    qr_command,
    qr_reset_command,
    scan_qr_and_markers,
    update_qr_confirmation,
    PersonDetection,
)


def make_person(distance_m):
    return PersonDetection(
        person_id="Person 1",
        bbox=(10, 20, 60, 120),
        center=(35, 70),
        width=50,
        height=100,
        confidence=0.9,
        distance=distance_m,
        horizontal_position="CENTER",
        vertical_position="CENTER",
    )


class VisionSafetyTests(unittest.TestCase):
    def test_no_person_is_safe(self):
        self.assertEqual(get_robot_status([]), "SAFE")
        self.assertEqual(arduino_safety_command(get_robot_status([])), "SAFE")

    def test_person_at_three_meters_is_safe(self):
        people = [make_person(3.0)]
        self.assertEqual(get_robot_status(people), "SAFE")
        self.assertEqual(arduino_safety_command(get_robot_status(people)), "SAFE")

    def test_person_at_one_and_a_half_meters_warns_but_is_arduino_safe(self):
        people = [make_person(1.5)]
        self.assertEqual(get_robot_status(people), "WARNING")
        self.assertEqual(arduino_safety_command(get_robot_status(people)), "SAFE")

    def test_exactly_one_meter_stops(self):
        people = [make_person(1.0)]
        self.assertEqual(get_robot_status(people), "STOP")
        self.assertEqual(arduino_safety_command(get_robot_status(people)), "STOP")

    def test_detected_person_without_pose_distance_fails_safe(self):
        people = [make_person(None)]
        self.assertEqual(get_robot_status(people), "PERSON_DETECTED")
        self.assertEqual(arduino_safety_command(get_robot_status(people)), "STOP")

    def test_distance_uses_calibrated_focal_length(self):
        expected = 1.76784 * 730.1040494938134 / 400
        self.assertAlmostEqual(calculate_distance(400, focal_length=730.1040494938134), expected)

    def test_nearest_valid_person_is_selected(self):
        people = [make_person(3.2), make_person(0.7), make_person(5.4)]
        nearest = find_nearest_person(people)
        self.assertIs(nearest, people[1])
        self.assertEqual(get_robot_status(people, nearest), "STOP")

    def test_heartbeat_repeats_for_unchanged_state(self):
        self.assertTrue(heartbeat_due(0.35, 0.0))
        self.assertFalse(heartbeat_due(0.34, 0.0))
        self.assertEqual(HEARTBEAT_INTERVAL_SECONDS, 0.35)

    def test_terminal_report_includes_safety_qr_and_heartbeat(self):
        output = StringIO()
        with redirect_stdout(output):
            log_frame_state(12.4, 2, True, "Person 1", 1.42, 4.66, "WARNING",
                            True, "ARUCO MARKER", "5", True, "SAFE", "QR:5", True, 0.1)
        report = output.getvalue()
        self.assertIn("People: 2", report)
        self.assertIn("Distance: 1.42 m / 4.66 ft", report)
        self.assertIn("Safety: WARNING", report)
        self.assertIn("QR: ARUCO MARKER 5 confirmed=True", report)
        self.assertIn("Arduino: SAFE (connected)", report)
        self.assertIn("Last command: QR:5", report)
        self.assertIn("Heartbeat: 0.10s ago", report)

    def test_qr_requires_two_matching_scans(self):
        recent = deque(maxlen=QR_CONFIRMATION_WINDOW)
        scans_without_code, qr_type, qr_value = update_qr_confirmation("7", "ARUCO MARKER", recent, 0, None, None)
        self.assertEqual((qr_type, qr_value), (None, None))
        scans_without_code, qr_type, qr_value = update_qr_confirmation("7", "ARUCO MARKER", recent, scans_without_code, qr_type, qr_value)
        self.assertEqual((qr_type, qr_value), ("ARUCO MARKER", "7"))

    def test_aruco_ids_one_and_ten_are_detected(self):
        qr_detector, marker_detector, enhancer = initialize_qr_detectors()
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        for marker_id in (1, 10):
            marker = cv2.aruco.generateImageMarker(dictionary, marker_id, 180)
            frame = np.full((260, 260, 3), 255, dtype=np.uint8)
            frame[40:220, 40:220] = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
            value, detection_type, boundary = scan_qr_and_markers(
                frame, 1, qr_detector, marker_detector, enhancer
            )
            self.assertEqual(value, str(marker_id))
            self.assertEqual(detection_type, "ARUCO MARKER")
            self.assertIsNotNone(boundary)

    def test_qrcode_detector_decodes_values_one_and_ten(self):
        encoder = cv2.QRCodeEncoder_create()
        qr_detector, marker_detector, enhancer = initialize_qr_detectors()
        for value in ("1", "10"):
            qr = encoder.encode(value)
            qr = cv2.resize(qr, (220, 220), interpolation=cv2.INTER_NEAREST)
            frame = np.full((320, 320, 3), 255, dtype=np.uint8)
            frame[50:270, 50:270] = cv2.cvtColor(qr, cv2.COLOR_GRAY2BGR)
            detected, detection_type, boundary = scan_qr_and_markers(
                frame, 1, qr_detector, marker_detector, enhancer
            )
            self.assertEqual(detected, value)
            self.assertEqual(detection_type, "QR CODE")
            self.assertIsNotNone(boundary)

    def test_qr_serial_commands_are_limited_to_one_through_ten(self):
        self.assertEqual(qr_command(1), "QR:1")
        self.assertEqual(qr_command(10), "QR:10")
        self.assertIsNone(qr_command(0))
        self.assertIsNone(qr_command(11))

    def test_confirmed_qr_stays_queued_until_sent(self):
        self.assertEqual(pending_qr_command("10", None), "QR:10")
        self.assertIsNone(pending_qr_command("10", "10"))
        self.assertIsNone(pending_qr_command(None, "10"))

    def test_qr_outputs_reset_only_after_confirmed_loss(self):
        self.assertIsNone(qr_reset_command("8", "8"))
        self.assertEqual(qr_reset_command(None, "8"), "RESET_QR")
        self.assertIsNone(qr_reset_command(None, None))


if __name__ == "__main__":
    unittest.main()
