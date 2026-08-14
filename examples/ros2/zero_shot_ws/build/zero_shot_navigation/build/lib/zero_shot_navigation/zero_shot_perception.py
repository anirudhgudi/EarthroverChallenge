#!/usr/bin/env python3
"""Zero-Shot Perception and Spatial Translation Node.

Subscribes to:
    /earth_rover/front/image_raw (sensor_msgs/Image)

Publishes:
    /local_costmap (nav_msgs/OccupancyGrid)
    /clip_similarity (std_msgs/Float32)

Requires:
    pip install ultralytics transformers torch opencv-python cv_bridge
"""

import math
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Float32, Header
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from cv_bridge import CvBridge
import cv2

try:
    import torch
    from ultralytics import YOLO
    from transformers import CLIPProcessor, CLIPModel
except ImportError as e:
    print(f"Failed to import ML libraries: {e}")
    # Handled gracefully if not installed

# Camera Intrinsics
# FOV D148° / H126° / V67°, effective focal length 2.72 mm
H_FOV_RAD = math.radians(126.0)
V_FOV_RAD = math.radians(67.0)
IMG_WIDTH = 1024
IMG_HEIGHT = 576

# Occupancy Grid settings (matches typical mini+ setup)
GRID_RESOLUTION = 0.05  # 5 cm per cell
GRID_WIDTH_M = 5.0      # 5x5 meters
GRID_HEIGHT_M = 5.0
GRID_CELL_WIDTH = int(GRID_WIDTH_M / GRID_RESOLUTION)
GRID_CELL_HEIGHT = int(GRID_HEIGHT_M / GRID_RESOLUTION)

class ZeroShotPerception(Node):
    def __init__(self):
        super().__init__('zero_shot_perception')
        self.bridge = CvBridge()
        
        self.declare_parameter('target_image_path', '')
        self.declare_parameter('yolo_model', 'yolov8n.pt')
        self.declare_parameter('clip_model', 'openai/clip-vit-base-patch32')

        yolo_model_path = self.get_parameter('yolo_model').value
        clip_model_name = self.get_parameter('clip_model').value
        target_image_path = self.get_parameter('target_image_path').value

        self.get_logger().info("Loading models... This may take a moment.")
        
        # Load YOLO
        self.yolo = YOLO(yolo_model_path)
        
        # Load CLIP
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model = CLIPModel.from_pretrained(clip_model_name).to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
        
        self.target_features = None
        if target_image_path:
            try:
                target_img = cv2.imread(target_image_path)
                target_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)
                inputs = self.clip_processor(images=target_img, return_tensors="pt").to(self.device)
                target_features = self.clip_model.get_image_features(**inputs)
                if hasattr(target_features, 'pooler_output'):
                    self.target_features = target_features.pooler_output
                else:
                    self.target_features = target_features
                self.target_features /= self.target_features.norm(dim=-1, keepdim=True)
                self.get_logger().info(f"Loaded target image: {target_image_path}")
            except Exception as e:
                self.get_logger().error(f"Failed to load target image: {e}")

        # Publishers
        self.costmap_pub = self.create_publisher(OccupancyGrid, '/local_costmap', 10)
        self.clip_pub = self.create_publisher(Float32, '/clip_similarity', 10)
        
        # Subscriber
        sensor_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )
        self.create_subscription(Image, '/earth_rover/front/image_raw', self.image_callback, sensor_qos)
        
        self.get_logger().info("Zero-Shot Perception Node started.")

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"CV Bridge error: {e}")
            return
            
        # 1. Run YOLO
        results = self.yolo(cv_image, verbose=False)
        boxes = results[0].boxes
        
        # 2. Update Occupancy Grid
        grid = self.create_empty_grid(msg.header.stamp)
        self.project_boxes_to_grid(boxes, grid, cv_image.shape)
        self.costmap_pub.publish(grid)
        
        # 3. Run CLIP (if target image exists)
        if self.target_features is not None:
            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            inputs = self.clip_processor(images=rgb_image, return_tensors="pt").to(self.device)
            image_features = self.clip_model.get_image_features(**inputs)
            if hasattr(image_features, 'pooler_output'):
                image_features = image_features.pooler_output
            image_features /= image_features.norm(dim=-1, keepdim=True)
            
            similarity = (image_features @ self.target_features.T).item()
            sim_msg = Float32()
            sim_msg.data = similarity
            self.clip_pub.publish(sim_msg)
            
            if similarity > 0.85: # Threshold for match
                self.get_logger().info(f"Target match found! Similarity: {similarity:.2f}")

    def create_empty_grid(self, stamp):
        grid = OccupancyGrid()
        grid.header = Header()
        grid.header.stamp = stamp
        grid.header.frame_id = 'earth_rover_front_camera'
        
        grid.info.resolution = GRID_RESOLUTION
        grid.info.width = GRID_CELL_WIDTH
        grid.info.height = GRID_CELL_HEIGHT
        
        # Origin is bottom center of the grid
        grid.info.origin.position.x = 0.0
        grid.info.origin.position.y = -GRID_WIDTH_M / 2.0
        grid.info.origin.position.z = 0.0
        grid.info.origin.orientation.w = 1.0
        
        # Initialize with free space (0) or unknown (-1)
        grid.data = [-1] * (GRID_CELL_WIDTH * GRID_CELL_HEIGHT)
        return grid

    def project_boxes_to_grid(self, boxes, grid, img_shape):
        data = np.array(grid.data).reshape((GRID_CELL_HEIGHT, GRID_CELL_WIDTH))
        # Clear known space to 0 (free)
        data.fill(0)
        
        h, w = img_shape[:2]
        
        for box in boxes:
            # box.xyxy format: [x1, y1, x2, y2]
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            
            # Simple heuristic projection using camera parameters
            # The lower the y2, the closer the object.
            # Convert pixel coordinates to angles using FOV
            
            # Center of the bounding box bottom edge
            px_x = (x1 + x2) / 2.0
            px_y = y2
            
            # Horizontal angle (from center)
            hfov_per_px = H_FOV_RAD / w
            angle = (px_x - w/2.0) * hfov_per_px
            
            # Vertical angle (from center). Simple flat ground assumption
            vfov_per_px = V_FOV_RAD / h
            # Angle relative to optical axis
            pitch = (px_y - h/2.0) * vfov_per_px
            
            # Assuming camera is mounted at some height, distance is a function of pitch
            # For simplicity without true height, we use a proportional scale
            # Max distance (top of screen) is say 5 meters. 
            distance = max(0.5, (h - px_y) / h * GRID_HEIGHT_M)
            
            # Polar to Cartesian
            obj_x = distance * math.cos(angle)
            obj_y = distance * math.sin(angle)
            
            # Map to grid indices
            grid_x = int(obj_x / GRID_RESOLUTION)
            grid_y = int((obj_y + GRID_WIDTH_M/2.0) / GRID_RESOLUTION)
            
            # Mark a small area as occupied (e.g. 3x3 cells)
            for i in range(-2, 3):
                for j in range(-2, 3):
                    gx = grid_x + i
                    gy = grid_y + j
                    if 0 <= gx < GRID_CELL_HEIGHT and 0 <= gy < GRID_CELL_WIDTH:
                        data[gx, gy] = 100 # Occupied
                        
        grid.data = data.flatten().tolist()

def main(args=None):
    rclpy.init(args=args)
    node = ZeroShotPerception()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
