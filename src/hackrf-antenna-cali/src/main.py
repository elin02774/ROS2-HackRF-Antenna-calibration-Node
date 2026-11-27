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

    iq_f32 = np.frombuffer(raw_data, dtype=np.int8).astype(np.float32) * (1.0 / 128.0)
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
        self.declare_parameter('antenna_link', 'antenna_link_1')
        self.declare_parameter('cmd', '/cmd_vel')
        self.declare_parameter('rssi_threshold', -26.5)
        self.declare_parameter('step', 10.0)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        cmd = self.get_parameter('cmd').get_parameter_value().string_value
        self.cmd_pub = self.create_publisher(Twist, cmd, 10)
        self.records = []

    def get_link_orientation(self):
        target_link = (
            self.get_parameter('target_link').get_parameter_value().string_value
        )
        antenna_link = (
            self.get_parameter('antenna_link').get_parameter_value().string_value
        )
        try:
            trans = self.tf_buffer.lookup_transform(
                target_link,
                antenna_link,
                rclpy.time.Time(),
            )
            return trans.transform.rotation
        except TransformException:
            return None

    def save_measurement(
        self,
        out_file,
        orientation,
        avg_rssi,
        vld,
        step,
        gain_lna,
        gain_vga,
    ):
        self.records.append({
            'orientation_w': None if orientation is None else float(orientation.w),
            'orientation_x': None if orientation is None else float(orientation.x),
            'orientation_y': None if orientation is None else float(orientation.y),
            'orientation_z': None if orientation is None else float(orientation.z),
            'avg_rssi': avg_rssi,
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

        sample_count = int(round(360.0 / step))
        hackrf = open_hackrf(frequency, LNA_GAIN, VGA_GAIN)
        try:
            for i in range(sample_count):
                rssi = read_rssi(hackrf, rssi_threshold)
                orientation = self.get_link_orientation()
                self.save_measurement(
                    out_file,
                    orientation,
                    rssi.avg,
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