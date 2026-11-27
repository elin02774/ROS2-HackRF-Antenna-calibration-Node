#!/usr/bin/env python3

import logging
import math
import time
from types import SimpleNamespace

from geometry_msgs.msg import Twist
import numpy as np
from pyhackrf2 import HackRF
from pythonjsonlogger import jsonlogger
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
)


def open_hackrf(frequency, lna_gain, vga_gain) -> HackRF:
    hackrf = HackRF()
    hackrf.sample_rate = SAMPLE_RATE
    hackrf.center_freq = frequency
    hackrf.lna_gain = lna_gain
    hackrf.vga_gain = vga_gain
    hackrf.amplifier_enable = False

    return hackrf


def read_rssi(frequency, rssi_threshold, lna_gain, vga_gain) -> SimpleNamespace:
    hackrf = open_hackrf(frequency, lna_gain, vga_gain)

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
        hackrf.close()

    raw_data = b''.join(raw_chunks)
    raw_data = raw_data[: len(raw_data) - (len(raw_data) % 2)]
    iq_int8 = np.frombuffer(raw_data, dtype=np.int8).astype(np.float32) / 128.0
    iq = iq_int8[::2] + 1j * iq_int8[1::2]

    if len(iq) == 0:
        raise RuntimeError('HackRF returned no IQ samples')

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

    noise_floor = np.median(frames_rssi)
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
    rssi.beacon_peaks = (
        len(beacon_peaks) if len(beacon_peaks) >= 30 else None
    )
    rssi.avg = np.median(beacon_peaks) if beacon_peaks else None
    rssi.min = np.min(beacon_peaks) if beacon_peaks else None
    rssi.max = np.max(beacon_peaks) if beacon_peaks else None
    rssi.vld = (
        rssi.beacon_peaks is not None
        and rssi.avg is not None
        and rssi.min is not None
        and rssi.max is not None
        and rssi.avg > rssi_threshold
    )
    rssi.ratio = (
        rssi.avg - rssi.noise_floor if rssi.noise_floor != 0 else None
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
        self.timer = self.create_timer(10.0, self.timer_callback)

    def get_link_orientation(self):
        target_link = (
            self.get_parameter('target_link').get_parameter_value().string_value
        )
        antenna_link = (
            self.get_parameter('antenna_link').get_parameter_value().string_value
        )
        try:
            now = rclpy.time.Time()
            trans = self.tf_buffer.lookup_transform(
                target_link,
                antenna_link,
                now,
            )
            return trans.transform.rotation
        except TransformException:
            return None

    @staticmethod
    def log_to_json(
        out_file,
        orientation,
        avg_rssi,
        vld,
        step,
        gain_lna,
        gain_vga,
    ):
        logger = logging.getLogger(f'json_logger_{out_file}')
        logger.setLevel(logging.INFO)
        logger.propagate = False

        if not logger.handlers:
            file_handler = logging.FileHandler(out_file, encoding='utf-8')
            formatter = jsonlogger.JsonFormatter(
                '%(asctime)s %(levelname)s %(message)s'
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        if orientation is None:
            logger.info(
                'measurement',
                extra={
                    'orientation_w': None,
                    'orientation_x': None,
                    'orientation_y': None,
                    'orientation_z': None,
                    'avg_rssi': float(avg_rssi) if avg_rssi is not None else None,
                    'vld': vld,
                    'step': step,
                    'gain_lna': gain_lna,
                    'gain_vga': gain_vga,
                },
            )
            return

        logger.info(
            'measurement',
            extra={
                'orientation_w': orientation.w,
                'orientation_x': orientation.x,
                'orientation_y': orientation.y,
                'orientation_z': orientation.z,
                'avg_rssi': float(avg_rssi) if avg_rssi is not None else None,
                'vld': vld,
                'step': step,
                'gain_lna': gain_lna,
                'gain_vga': gain_vga,
            },
        )

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
            rclpy.spin_once(self, timeout_sec=0.1)

        twist_msg.angular.z = 0.0
        self.cmd_pub.publish(twist_msg)

    def timer_callback(self):
        self.timer.cancel()

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
            (360.0 / step) % 1.0,
            0.0,
            abs_tol=1e-9,
        ):
            raise ValueError('step must divide 360 degrees evenly')

        sample_count = int(round(360.0 / step))
        for i in range(sample_count):
            rssi = read_rssi(
                frequency,
                rssi_threshold,
                LNA_GAIN,
                VGA_GAIN,
            )
            orientation = self.get_link_orientation()
            self.log_to_json(
                out_file,
                orientation,
                rssi.avg,
                rssi.vld,
                i * step,
                LNA_GAIN,
                VGA_GAIN,
            )
            self.rotate(math.radians(step), speed_w=0.5)

        rssi = read_rssi(frequency, rssi_threshold, LNA_GAIN, VGA_GAIN)
        self.get_logger().info(
            f'Average RSSI: {rssi.avg}, '
            f'Noise Floor: {rssi.noise_floor}, '
            f'Beacon Peaks: {rssi.beacon_peaks}, '
            f'Valid: {rssi.vld}, '
            f'SNR: {rssi.ratio}'
        )


def main():
    rclpy.init()
    node = AntennaCalibrationNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
