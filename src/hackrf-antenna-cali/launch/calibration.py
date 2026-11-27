from launch import LaunchDescription
from launch.actions import Node

def generate_launch_description():
    ld = LaunchDescription()
    calibration_node = Node(
        package='hackrf-antenna-cali',
        executable='calibration_node',
        name='antenna_calibration_node',
        output='screen',
        parameters=[
            {'out_file': 'lookup.json'},
            {'frequency': 2480.00e6},
            {'target_link': 'odom'},
            {'antenna_link': 'antenna_link_1'},
            {'cmd': '/cmd_vel'},
            {'rssi_threshold': -26.5},
            {'step': 10.0},
        ],
    )
    ld.add_action(calibration_node)
    return ld   