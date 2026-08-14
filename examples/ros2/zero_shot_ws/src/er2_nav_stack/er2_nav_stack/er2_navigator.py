import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import collections
import os
import time
import asyncio
import json
from nav_msgs.msg import Odometry
import math
from google import genai
from google.genai import types

class ER2Navigator(Node):
    def __init__(self):
        super().__init__('er2_navigator')
        
        self.declare_parameter('api_timer_period', 1.0)
        self.declare_parameter('watchdog_timeout', 1.25)
        
        api_timer_period = self.get_parameter('api_timer_period').value
        self.watchdog_timeout = self.get_parameter('watchdog_timeout').value
        
        self.subscription = self.create_subscription(
            Image,
            '/earthrover/front_camera/image_raw',
            self.image_callback,
            10)
            
        self.odom_sub = self.create_subscription(
            Odometry,
            '/odometry/filtered',
            self.odom_callback,
            10)
            
        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        
        self.bridge = CvBridge()
        
        # Buffer to keep 1 frame every 0.5 seconds, max 10 frames
        self.frame_buffer = collections.deque(maxlen=10)
        self.last_frame_time = 0.0
        
        # Current state
        self.current_speed = 0.0
        self.current_heading = 0.0
        
        self.api_key = os.getenv('GEMINI_API_KEY')
        if not self.api_key:
            self.get_logger().error('GEMINI_API_KEY environment variable is not set!')
            
        # Initialize Gemini Client
        self.client = genai.Client(api_key=self.api_key) if self.api_key else None
        self.cached_content = None
        
        # Tracking API latency and watchdog
        self.last_api_success_time = time.time()
        self.is_api_running = False
        
        # Timers
        self.api_timer = self.create_timer(api_timer_period, self.api_timer_callback)
        self.watchdog_timer = self.create_timer(0.1, self.watchdog_callback)
        
        self.get_logger().info('ER2 Navigator Node initialized.')

    def odom_callback(self, msg):
        self.current_speed = msg.twist.twist.linear.x
        
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.current_heading = math.degrees(math.atan2(siny_cosp, cosy_cosp))

    def image_callback(self, msg):
        current_time = time.time()
        # Capture exactly 1 frame every 0.5 seconds
        if current_time - self.last_frame_time >= 0.5:
            try:
                cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
                # Encode as JPEG
                _, buffer = cv2.imencode('.jpg', cv_image)
                # Convert to part for Gemini
                image_bytes = buffer.tobytes()
                
                self.frame_buffer.append({
                    'mime_type': 'image/jpeg',
                    'data': image_bytes,
                    'timestamp': current_time
                })
                self.last_frame_time = current_time
                self.get_logger().debug(f'Appended frame to buffer. Buffer size: {len(self.frame_buffer)}')
            except Exception as e:
                self.get_logger().error(f'Error processing image: {e}')

    def api_timer_callback(self):
        if not self.client:
            return
            
        if self.is_api_running:
            self.get_logger().debug('API call already in progress, skipping this timer cycle.')
            return
            
        if len(self.frame_buffer) < 2:
            self.get_logger().debug('Not enough frames in buffer to query API.')
            return
            
        # Run API call asynchronously so we don't block the ROS executor
        self.is_api_running = True
        import threading
        t = threading.Thread(target=self.query_gemini)
        t.start()

    def query_gemini(self):
        try:
            frames = list(self.frame_buffer)
            
            # Prepare contents for caching
            cache_contents = []
            for frame in frames:
                cache_contents.append(
                    types.Part.from_bytes(data=frame['data'], mime_type=frame['mime_type'])
                )
                
            prompt = (
                f"Current State: Speed {self.current_speed:.2f} m/s, Heading: {self.current_heading:.1f} degrees.\n"
                "You are an autonomous Earth Rover. You ARE the rover. Your goal is to move forward safely while avoiding obstacles.\n"
                "Safety-first: Use conservative speeds. If unsure, stop (linear_x: 0, angular_z: 0).\n"
                "Safe speeds: default linear speed 0.3 to 0.5, angular speed 0.3 to 0.4. Never exceed 0.8.\n"
                "Analyze the provided temporal sequence of front-facing camera frames.\n"
                "Output ONLY a valid JSON object containing linear_x (float) and angular_z (float).\n"
                "Do not include markdown formatting or explanations."
            )
            
            model_name = 'gemini-3.6-flash'
            
            # Explicit Context Caching:
            # We delete the previous cache if it exists to replace it with the new sliding window
            if self.cached_content:
                try:
                    self.client.caches.delete(name=self.cached_content.name)
                except Exception as e:
                    self.get_logger().debug(f'Failed to delete old cache (it might have expired): {e}')
                    
            # Create a new cache with the video frames
            self.cached_content = self.client.caches.create(
                model=model_name,
                contents=cache_contents
            )
            
            response = self.client.models.generate_content(
                model=model_name,
                contents=[prompt],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    cached_content=self.cached_content.name,
                )
            )
            
            response_text = response.text
            
            try:
                # Sometimes it includes markdown ```json ... ```
                response_text = response_text.replace('```json', '').replace('```', '').strip()
                data = json.loads(response_text)
                
                linear_x = float(data.get('linear_x', 0.0))
                angular_z = float(data.get('angular_z', 0.0))
                
                # Clamp values
                linear_x = max(-1.0, min(1.0, linear_x))
                angular_z = max(-1.0, min(1.0, angular_z))
                
                twist = Twist()
                twist.linear.x = linear_x
                twist.angular.z = angular_z
                
                self.publisher.publish(twist)
                self.last_api_success_time = time.time()
                
            except json.JSONDecodeError:
                self.get_logger().error(f'Failed to parse JSON response: {response_text}')
            
        except Exception as e:
            self.get_logger().error(f'Gemini API Error: {e}')
        finally:
            self.is_api_running = False

    def watchdog_callback(self):
        current_time = time.time()
        # If API is taking too long or hasn't succeeded in watchdog_timeout seconds
        if current_time - self.last_api_success_time > self.watchdog_timeout:
            # Publish zero velocity
            twist = Twist()
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            self.publisher.publish(twist)
            # self.get_logger().warn('Watchdog triggered! Stopping rover.')

def main(args=None):
    rclpy.init(args=args)
    node = ER2Navigator()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
