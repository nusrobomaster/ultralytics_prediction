#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import random
import time
import math

class GimbalPublisher(Node):
    def __init__(self):
        super().__init__('gimbal_publisher')
        self.publisher_ = self.create_publisher(Float32MultiArray, 'gimbal_angles', 10)
        self.timer = self.create_timer(5.0, self.publish_gimbal_angles)
        self.get_logger().info('Gimbal Publisher Node has been started.')

    def publish_gimbal_angles(self):
        pitch = random.uniform(math.pi/2, -math.pi/2)
        yaw = random.uniform(-math.pi, math.pi)
        msg = Float32MultiArray()
        msg.data = [pitch, yaw]
        self.publisher_.publish(msg)
        self.get_logger().info(f'Published Pitch: {pitch}, Yaw: {yaw}')

def main(args=None):
    rclpy.init(args=args)
    node = GimbalPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
