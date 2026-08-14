import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    pkg_dir = get_package_share_directory('er2_nav_stack')
    ekf_config_file = os.path.join(pkg_dir, 'config', 'ekf.yaml')

    return LaunchDescription([
        Node(
            package='er2_nav_stack',
            executable='frodobots_ekf_bridge',
            name='frodobots_ekf_bridge',
            output='screen'
        ),
        Node(
            package='robot_localization',
            executable='ekf_node',
            name='ekf_filter_node',
            output='screen',
            parameters=[ekf_config_file]
        ),
        Node(
            package='er2_nav_stack',
            executable='er2_navigator',
            name='er2_navigator',
            output='screen'
        )
    ])
