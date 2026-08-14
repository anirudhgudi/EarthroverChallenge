import os
import sys
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node

def generate_launch_description():
    pkg_dir = get_package_share_directory('zero_shot_navigation')
    nav2_params = os.path.join(pkg_dir, 'config', 'nav2_params.yaml')

    return LaunchDescription([
        # Start the Earth Rover SDK Bridge
        ExecuteProcess(
            cmd=[sys.executable, '/home/mango/Earthrover/earth-rovers-sdk/examples/ros2/earth_rover_bridge.py'],
            output='screen'
        ),

        # Start Perception Node
        Node(
            package='zero_shot_navigation',
            executable='zero_shot_perception',
            name='zero_shot_perception',
            output='screen',
            parameters=[
                {'yolo_model': 'yolov8n.pt'},
                {'clip_model': 'openai/clip-vit-base-patch32'},
                {'target_image_path': '/home/mango/Earthrover/goal.jpg'}
            ]
        ),

        # Start Custom Trajectory Planner
        Node(
            package='zero_shot_navigation',
            executable='zero_shot_planner',
            name='zero_shot_planner',
            output='screen',
            parameters=[
                {'target_lat': 30.482748},
                {'target_lng': 114.302650}
            ]
        )
    ])
