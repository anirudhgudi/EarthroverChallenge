import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu, MagneticField, NavSatFix
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Quaternion, Twist, TransformStamped
import requests
import math
import time
from rclpy.time import Time
from rclpy.qos import QoSProfile, qos_profile_sensor_data

class FrodobotsEKFBridge(Node):
    def __init__(self):
        super().__init__('frodobots_ekf_bridge')
        
        self.declare_parameter('poll_rate', 20.0) # Hz
        poll_rate = self.get_parameter('poll_rate').value
        
        self.imu_pub = self.create_publisher(Imu, '/frodobot/imu', qos_profile_sensor_data)
        self.mag_pub = self.create_publisher(MagneticField, '/frodobot/mag', qos_profile_sensor_data)
        self.odom_pub = self.create_publisher(Odometry, '/frodobot/odom', 10)
        self.gps_pub = self.create_publisher(NavSatFix, '/frodobot/gps', qos_profile_sensor_data)
        
        # Kinematics variables
        self.wheel_diameter = 0.095
        self.wheelbase = 0.160
        
        self.odom_x = 0.0
        self.odom_y = 0.0
        self.odom_theta = 0.0
        self.last_rpm_timestamp = None
        
        self.timer = self.create_timer(1.0 / poll_rate, self.poll_api)
        
        self.get_logger().info('Frodobots EKF Bridge initialized.')

    def euler_to_quaternion(self, roll, pitch, yaw):
        qx = math.sin(roll/2) * math.cos(pitch/2) * math.cos(yaw/2) - math.cos(roll/2) * math.sin(pitch/2) * math.sin(yaw/2)
        qy = math.cos(roll/2) * math.sin(pitch/2) * math.cos(yaw/2) + math.sin(roll/2) * math.cos(pitch/2) * math.sin(yaw/2)
        qz = math.cos(roll/2) * math.cos(pitch/2) * math.sin(yaw/2) - math.sin(roll/2) * math.sin(pitch/2) * math.cos(yaw/2)
        qw = math.cos(roll/2) * math.cos(pitch/2) * math.cos(yaw/2) + math.sin(roll/2) * math.sin(pitch/2) * math.sin(yaw/2)
        return Quaternion(x=qx, y=qy, z=qz, w=qw)

    def process_timestamp(self, ts):
        # Assumes timestamp is in seconds (float) or milliseconds. 
        # Usually epoch is large float. If > 1e10, it's ms. Let's handle both.
        if ts > 1e10:
            ts = ts / 1000.0
        sec = int(ts)
        nanosec = int((ts - sec) * 1e9)
        return Time(seconds=sec, nanoseconds=nanosec).to_msg()

    def poll_api(self):
        try:
            response = requests.get('http://localhost:8000/data', timeout=0.2)
            if response.status_code == 200:
                data = response.json()
                self.process_data(data)
        except Exception as e:
            # Silently fail on timeout or connection error to avoid spam, or log debug
            self.get_logger().debug(f'API error: {e}')

    def process_data(self, data):
        # GPS
        if 'latitude' in data and 'longitude' in data and 'timestamp' in data:
            msg = NavSatFix()
            msg.header.stamp = self.process_timestamp(data['timestamp'])
            msg.header.frame_id = 'base_link'
            msg.latitude = data['latitude']
            msg.longitude = data['longitude']
            self.gps_pub.publish(msg)

        # IMU (accels, gyros)
        accels = data.get('accels', [])
        gyros = data.get('gyros', [])
        
        # We might have batches. For simplicity, pair them or process accels as IMU msgs.
        # If gyros are 1Hz and accels are 100Hz, we can broadcast an IMU msg for each accel,
        # using the latest gyro.
        
        latest_gyro = None
        if gyros:
            latest_gyro = gyros[-1] # [x, y, z, ts]

        for a in accels:
            if len(a) >= 4:
                msg = Imu()
                msg.header.stamp = self.process_timestamp(a[3])
                msg.header.frame_id = 'base_link'
                
                msg.linear_acceleration.x = float(a[0])
                msg.linear_acceleration.y = float(a[1])
                msg.linear_acceleration.z = float(a[2])
                
                if latest_gyro and len(latest_gyro) >= 4:
                    msg.angular_velocity.x = float(latest_gyro[0])
                    msg.angular_velocity.y = float(latest_gyro[1])
                    msg.angular_velocity.z = float(latest_gyro[2])
                
                # Covariances can be left as 0, EKF will use parameters if specified
                self.imu_pub.publish(msg)

        # Magnetometer
        mags = data.get('mags', [])
        for m in mags:
            if len(m) >= 4:
                msg = MagneticField()
                msg.header.stamp = self.process_timestamp(m[3])
                msg.header.frame_id = 'base_link'
                msg.magnetic_field.x = float(m[0])
                msg.magnetic_field.y = float(m[1])
                msg.magnetic_field.z = float(m[2])
                self.mag_pub.publish(msg)

        # Odometry (rpms)
        rpms = data.get('rpms', [])
        circumference = math.pi * self.wheel_diameter
        
        for r in rpms:
            if len(r) >= 5:
                # Assume layout: [FL, FR, RL, RR, ts]
                # If it's left/right layout: [L1, R1, L2, R2] or something.
                # Standard is usually FL, FR, RL, RR
                rpm_fl, rpm_fr, rpm_rl, rpm_rr, ts = r[0], r[1], r[2], r[3], r[4]
                
                rpm_left = (rpm_fl + rpm_rl) / 2.0
                rpm_right = (rpm_fr + rpm_rr) / 2.0
                
                v_left = (rpm_left / 60.0) * circumference
                v_right = (rpm_right / 60.0) * circumference
                
                v_linear = (v_left + v_right) / 2.0
                v_angular = (v_right - v_left) / self.wheelbase
                
                if self.last_rpm_timestamp is not None:
                    dt = ts - self.last_rpm_timestamp
                    if dt > 0 and dt < 1.0: # Sanity check
                        # Integrate
                        self.odom_theta += v_angular * dt
                        self.odom_x += v_linear * math.cos(self.odom_theta) * dt
                        self.odom_y += v_linear * math.sin(self.odom_theta) * dt
                
                self.last_rpm_timestamp = ts
                
                odom = Odometry()
                odom.header.stamp = self.process_timestamp(ts)
                odom.header.frame_id = 'odom'
                odom.child_frame_id = 'base_link'
                
                odom.pose.pose.position.x = self.odom_x
                odom.pose.pose.position.y = self.odom_y
                odom.pose.pose.position.z = 0.0
                odom.pose.pose.orientation = self.euler_to_quaternion(0, 0, self.odom_theta)
                
                odom.twist.twist.linear.x = v_linear
                odom.twist.twist.angular.z = v_angular
                
                self.odom_pub.publish(odom)

def main(args=None):
    rclpy.init(args=args)
    node = FrodobotsEKFBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
