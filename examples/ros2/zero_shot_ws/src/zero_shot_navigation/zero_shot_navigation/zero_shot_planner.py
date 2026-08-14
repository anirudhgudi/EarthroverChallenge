#!/usr/bin/env python3
"""Zero-Shot Custom Trajectory Planner.

Subscribes to:
    /local_costmap (nav_msgs/OccupancyGrid)
    /clip_similarity (std_msgs/Float32)

Publishes:
    /cmd_vel (geometry_msgs/Twist)
"""

import math
import time
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Float32
from geometry_msgs.msg import Twist
from sensor_msgs.msg import NavSatFix

class ZeroShotPlanner(Node):
    def __init__(self):
        super().__init__('zero_shot_planner')
        
        # State variables
        self.target_found = False
        self.obstacle_ahead = False
        self.similarity_score = 0.0
        self.current_lat = None
        self.current_lng = None
        self.current_heading = None
        
        # Parameters
        self.declare_parameter('target_lat', 0.0)
        self.declare_parameter('target_lng', 0.0)
        self.target_lat = self.get_parameter('target_lat').value
        self.target_lng = self.get_parameter('target_lng').value
        
        # Tuning parameters
        self.similarity_threshold = 0.85
        self.forward_speed = 0.3
        self.turn_speed = 0.5
        self.safe_distance_m = 1.0 # Stop if obstacle is within 1 meter ahead
        
        # Subscriptions
        self.create_subscription(OccupancyGrid, '/local_costmap', self.costmap_callback, 10)
        self.create_subscription(Float32, '/clip_similarity', self.similarity_callback, 10)
        self.create_subscription(NavSatFix, '/earth_rover/gps', self.gps_callback, 10)
        self.create_subscription(Float32, '/earth_rover/heading', self.heading_callback, 10)
        
        # Publisher
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # Control Loop (10 Hz)
        self.create_timer(0.1, self.control_loop)
        
        self.get_logger().info("Zero-Shot Custom Planner Node started.")

    def gps_callback(self, msg):
        self.current_lat = msg.latitude
        self.current_lng = msg.longitude
        
    def heading_callback(self, msg):
        self.current_heading = msg.data

    def similarity_callback(self, msg):
        self.similarity_score = msg.data
        if self.similarity_score > self.similarity_threshold:
            if not self.target_found:
                self.get_logger().info(f"TARGET MATCH FOUND ({self.similarity_score:.2f})! Stopping.")
            self.target_found = True
        else:
            self.target_found = False

    def costmap_callback(self, msg):
        # The costmap origin is at the bottom center (the camera's position)
        # We need to check the cells directly in front of the robot.
        
        res = msg.info.resolution
        width = msg.info.width
        height = msg.info.height
        
        # Find grid indices for the "ahead" region
        # Robot is at x = width / 2, y = 0
        center_x = width // 2
        
        # We check a corridor in front of the robot
        # Corridor width: e.g. 0.4 meters total (0.2m each side)
        cells_x_half = int((0.2) / res)
        
        # Corridor length: safe_distance_m
        cells_y_max = int(self.safe_distance_m / res)
        
        obstacle_detected = False
        
        for y in range(min(cells_y_max, height)):
            for x in range(center_x - cells_x_half, center_x + cells_x_half):
                if 0 <= x < width:
                    # Indexing: row-major order
                    idx = y * width + x
                    if msg.data[idx] > 50: # Threshold for occupied
                        obstacle_detected = True
                        break
            if obstacle_detected:
                break
                
        self.obstacle_ahead = obstacle_detected

    def calculate_bearing(self, lat1, lng1, lat2, lng2):
        dLon = math.radians(lng2 - lng1)
        lat1 = math.radians(lat1)
        lat2 = math.radians(lat2)
        y = math.sin(dLon) * math.cos(lat2)
        x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dLon)
        brng = math.atan2(y, x)
        brng = math.degrees(brng)
        return (brng + 360) % 360

    def calculate_distance(self, lat1, lng1, lat2, lng2):
        return math.hypot(lat2 - lat1, lng2 - lng1)

    def control_loop(self):
        cmd = Twist()
        
        if self.target_found:
            # Target reached (CLIP matched)
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
        elif self.obstacle_ahead:
            # Obstacle avoidance overrides GPS
            cmd.linear.x = 0.0
            cmd.angular.z = self.turn_speed # Turn left to avoid
        else:
            # Safe to move forward, check GPS guidance
            if self.target_lat != 0.0 and self.target_lng != 0.0 and self.current_lat is not None and self.current_heading is not None:
                dist = self.calculate_distance(self.current_lat, self.current_lng, self.target_lat, self.target_lng)
                
                if dist < 0.0002:
                    # Within ~ meters
                    # We are at the GPS location. Spin slowly to let CLIP find the target image.
                    cmd.linear.x = 0.0
                    cmd.angular.z = 0.2
                    self.get_logger().info(f"GPS Goal Reached (Error: {dist:.5f}). Scanning for image...", throttle_duration_sec=2.0)
                else:
                    target_bearing = self.calculate_bearing(self.current_lat, self.current_lng, self.target_lat, self.target_lng)
                    diff = (target_bearing - self.current_heading + 180) % 360 - 180
                    
                    self.get_logger().info(f"GPS Dist: {dist:.5f}, Bearing: {target_bearing:.1f}, Heading: {self.current_heading:.1f}, Diff: {diff:.1f}", throttle_duration_sec=1.0)
                    
                    if diff > 30:
                        # Turn Right
                        cmd.linear.x = 0.0 # Stop forward motion to turn sharply
                        cmd.angular.z = -self.turn_speed
                    elif diff < -30:
                        # Turn Left
                        cmd.linear.x = 0.0 # Stop forward motion to turn sharply
                        cmd.angular.z = self.turn_speed
                    else:
                        # On course
                        cmd.linear.x = self.forward_speed
                        # Add a scanning wobble to look sideways while moving
                        cmd.angular.z = math.sin(time.time() * 1.5) * 0.4
            else:
                # No GPS target set or no GPS lock yet, just explore forward
                cmd.linear.x = self.forward_speed
                cmd.angular.z = 0.0
                
        self.cmd_pub.publish(cmd)

def main(args=None):
    rclpy.init(args=args)
    node = ZeroShotPlanner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
