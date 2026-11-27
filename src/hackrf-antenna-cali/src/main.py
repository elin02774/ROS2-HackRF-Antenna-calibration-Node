#!/usr/bin/env python3

import json
import math
import threading
import time
from types import SimpleNamespace

from geometry_msgs.msg import Twist
import numpy as np
from pyhackrf2 import HackRF
import rclpy
import rclpy.node
from scipy import signal
from tf2_ros import Buffer, TransformException, TransformListener

SAMPLE_RATE = 8e6
MEASURE_SECONDS = 3.0
LNA_GAIN = 40
VGA_GAIN = 24

sos_filter = signal.butter(
    5,
    0.75e6,
    btype='low',
    fs=SAMPLE_RATE,
    output='sos',
).astype(np.float32)


def open_hackrf(frequency, lna_gain, vga_gain) -> HackRF:
    hackrf = HackRF()
    hackrf.sample_rate = SAMPLE_RATE
    hackrf.center_freq = frequency - 2.0e6
    hackrf.lna_gain = lna_gain
    hackrf.vga_gain = vga_gain
    hackrf.amplifier_enable = False
    return hackrf


def read_rssi(hackrf: HackRF, rssi_threshold) -> SimpleNamespace:
    raw_chunks = []

    def rx_callback(data: bytes):
        raw_chunks.append(bytes(data))
        return False

    started = False
    try:
        hackrf.start_rx(rx_callback)
        started = True
        time.sleep(MEASURE_SECONDS)
    finally:
        if started:
            hackrf.stop_rx()

    raw_data = b''.join(raw_chunks)
    raw_data = raw_data[: len(raw_data) - (len(raw_data) % 8)]
    if not raw_data:
        raise RuntimeError('HackRF returned no IQ samples')

    iq_f32 = (
        np.frombuffer(raw_data, dtype=np.int8).astype(np.float32)
        * (1.0 / 128.0)
    )
    iq = iq_f32.view(np.complex64)
    iq4 = iq.reshape(-1, 4)
    iq4[:, 1] *= -1j
    iq4[:, 2] *= -1.0
    iq4[:, 3] *= 1j

    filtered = signal.sosfilt(sos_filter, iq)

    window_size = int(100e-6 * SAMPLE_RATE)
    n_windows = len(filtered) // window_size
    if n_windows == 0:
        raise RuntimeError(
            'HackRF measurement was shorter than one analysis window'
        )
    frames = filtered[: n_windows * window_size].reshape(n_windows, window_size)

    frame_powers = np.mean(np.abs(frames) ** 2, axis=1)
    frames_rssi = 10 * np.log10(frame_powers + 1e-12)

    noise_floor = float(np.median(frames_rssi))
    above_threshold = frames_rssi > (noise_floor + 10)

    padded = np.concatenate(([False], above_threshold, [False]))
    starts = np.where(padded[1:] & ~padded[:-1])[0]
    ends = np.where(~padded[1:] & padded[:-1])[0]

    beacon_peaks = []
    for start, end in zip(starts, ends):
        duration_us = (end - start) * 100
        if 100 <= duration_us <= 600:
            beacon_peaks.append(np.max(frames_rssi[start:end]))

    rssi = SimpleNamespace()
    rssi.noise_floor = noise_floor
    rssi.beacon_peaks = len(beacon_peaks) if len(beacon_peaks) >= 30 else None
    rssi.avg = float(np.median(beacon_peaks)) if beacon_peaks else None
    rssi.min = float(np.min(beacon_peaks)) if beacon_peaks else None
    rssi.max = float(np.max(beacon_peaks)) if beacon_peaks else None
    rssi.vld = bool(
        rssi.beacon_peaks is not None
        and rssi.avg is not None
        and rssi.min is not None
        and rssi.max is not None
        and rssi.avg > rssi_threshold
    )
    rssi.ratio = (
        float(rssi.avg - rssi.noise_floor) if rssi.avg is not None else None
    )

    return rssi


class AntennaCalibrationNode(rclpy.node.Node):
    def __init__(self):
        super().__init__('antenna_calibration_node')
        self.get_logger().info('Antenna Calibration Node has been started.')

        self.declare_parameter('out_file', 'lookup.json')
        self.declare_parameter('frequency', 2480.00e6)
        self.declare_parameter('target_link', 'odom')
        self.declare_parameter('base_link', 'base_link')
        self.declare_parameter('antenna_link', 'antenna_link_1')
        self.declare_parameter('beacon_distance', 2.0)
        self.declare_parameter('cmd', '/cmd_vel')
        self.declare_parameter('rssi_threshold', -26.5)
        self.declare_parameter('step', 10.0)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        cmd = self.get_parameter('cmd').get_parameter_value().string_value
        self.cmd_pub = self.create_publisher(Twist, cmd, 10)
        self.records = []

    @staticmethod
    def quat_to_yaw(q) -> float:
        return math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z),
        )

    def get_transform(self, target_frame: str, source_frame: str):
        try:
            trans = self.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                rclpy.time.Time(),
            )
            return trans.transform
        except TransformException:
            return None

    @staticmethod
    def compensate_offset(avg_rssi, ant_tf, beacon_xy, ref_distance):
        if ant_tf is None:
            return avg_rssi, None, None, None

        ax = ant_tf.translation.x
        ay = ant_tf.translation.y
        yaw = AntennaCalibrationNode.quat_to_yaw(ant_tf.rotation)

        dx = beacon_xy[0] - ax
        dy = beacon_xy[1] - ay
        d_true = max(math.hypot(dx, dy), 1e-6)

        fspl_corr_db = 20.0 * math.log10(d_true / ref_distance)
        corrected_rssi = (
            None if avg_rssi is None else float(avg_rssi + fspl_corr_db)
        )

        bearing_odom = math.atan2(dy, dx)
        rel_angle_rad = math.atan2(
            math.sin(bearing_odom - yaw),
            math.cos(bearing_odom - yaw),
        )
        rel_angle_deg = math.degrees(rel_angle_rad)

        return (
            corrected_rssi,
            float(rel_angle_deg),
            float(d_true),
            float(fspl_corr_db),
        )

    def save_measurement(
        self,
        out_file,
        ant_tf,
        avg_rssi,
        corrected_rssi,
        rel_angle_deg,
        distance_m,
        fspl_corr_db,
        vld,
        step,
        gain_lna,
        gain_vga,
    ):
        orientation = None if ant_tf is None else ant_tf.rotation
        self.records.append({
            'orientation_w': None if orientation is None else float(orientation.w),
            'orientation_x': None if orientation is None else float(orientation.x),
            'orientation_y': None if orientation is None else float(orientation.y),
            'orientation_z': None if orientation is None else float(orientation.z),
            'avg_rssi': avg_rssi,
            'corrected_rssi': corrected_rssi,
            'rel_angle_deg': rel_angle_deg,
            'distance_m': distance_m,
            'fspl_corr_db': fspl_corr_db,
            'vld': vld,
            'step': float(step),
            'gain_lna': gain_lna,
            'gain_vga': gain_vga,
        })
        with open(out_file, 'w', encoding='utf-8') as f:
            json.dump(self.records, f, indent=2)

    def rotate(self, angle_rad: float, speed_w: float = 0.5):
        if speed_w <= 0:
            raise ValueError('speed_w must be greater than zero')

        twist_msg = Twist()
        twist_msg.angular.z = speed_w if angle_rad > 0 else -speed_w
        duration = abs(angle_rad) / speed_w
        end_time = self.get_clock().now() + rclpy.duration.Duration(
            seconds=duration
        )

        while self.get_clock().now() < end_time:
            self.cmd_pub.publish(twist_msg)
            time.sleep(0.05)

        twist_msg.angular.z = 0.0
        self.cmd_pub.publish(twist_msg)
        time.sleep(0.3)

    def run_calibration(self):
        out_file = (
            self.get_parameter('out_file').get_parameter_value().string_value
        )
        frequency = (
            self.get_parameter('frequency').get_parameter_value().double_value
        )
        target_link = (
            self.get_parameter('target_link').get_parameter_value().string_value
        )
        base_link = (
            self.get_parameter('base_link').get_parameter_value().string_value
        )
        antenna_link = (
            self.get_parameter('antenna_link')
            .get_parameter_value()
            .string_value
        )
        beacon_distance = (
            self.get_parameter('beacon_distance')
            .get_parameter_value()
            .double_value
        )
        rssi_threshold = (
            self.get_parameter('rssi_threshold')
            .get_parameter_value()
            .double_value
        )
        step = self.get_parameter('step').get_parameter_value().double_value

        if step <= 0 or not math.isclose(
            360.0 / step,
            round(360.0 / step),
            abs_tol=1e-6,
        ):
            raise ValueError('step must divide 360 degrees evenly')
        if beacon_distance <= 0:
            raise ValueError('beacon_distance must be greater than zero')

        start_tf = self.get_transform(target_link, base_link)
        if start_tf is None:
            raise RuntimeError(f'No TF between {target_link} and {base_link}')

        start_yaw = self.quat_to_yaw(start_tf.rotation)
        beacon_xy = (
            start_tf.translation.x + beacon_distance * math.cos(start_yaw),
            start_tf.translation.y + beacon_distance * math.sin(start_yaw),
        )

        sample_count = int(round(360.0 / step))
        hackrf = open_hackrf(frequency, LNA_GAIN, VGA_GAIN)
        try:
            for i in range(sample_count):
                rssi = read_rssi(hackrf, rssi_threshold)
                ant_tf = self.get_transform(target_link, antenna_link)
                (
                    corrected_rssi,
                    rel_angle_deg,
                    distance_m,
                    fspl_corr_db,
                ) = self.compensate_offset(
                    rssi.avg,
                    ant_tf,
                    beacon_xy,
                    beacon_distance,
                )
                self.save_measurement(
                    out_file,
                    ant_tf,
                    rssi.avg,
                    corrected_rssi,
                    rel_angle_deg,
                    distance_m,
                    fspl_corr_db,
                    rssi.vld,
                    i * step,
                    LNA_GAIN,
                    VGA_GAIN,
                )
                self.rotate(math.radians(step), speed_w=0.5)

            rssi = read_rssi(hackrf, rssi_threshold)
            self.get_logger().info(
                f'Average RSSI: {rssi.avg}, '
                f'Noise Floor: {rssi.noise_floor}, '
                f'Beacon Peaks: {rssi.beacon_peaks}, '
                f'Valid: {rssi.vld}, '
                f'SNR: {rssi.ratio}'
            )
        finally:
            hackrf.close()
            hackrf._device_opened = False


def main():
    rclpy.init()
    node = AntennaCalibrationNode()
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()
    try:
        time.sleep(1.0)
        node.run_calibration()
    finally:
        rclpy.shutdown()
        spin_thread.join(timeout=1.0)
        node.destroy_node()


if __name__ == '__main__':
    main()