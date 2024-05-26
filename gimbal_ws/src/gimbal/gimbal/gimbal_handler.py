from rclpy.node import Node
from std_msgs.msg import Float32MultiArray


class Gimbal:
    def __init__(self, image_width, image_height, HFOV, VFOV):
        self.image_width = image_width
        self.image_height = image_height
        self.HFOV = HFOV
        self.VFOV = VFOV

    def calculate_gimbal_adjustment(self, bbox):
        bbox_center_x = (bbox[0] + bbox[2]) / 2
        bbox_center_y = (bbox[1] + bbox[3]) / 2
        image_center_x = self.image_width / 2
        image_center_y = self.image_height / 2
        delta_x = bbox_center_x - image_center_x
        delta_y = -(bbox_center_y - image_center_y)
        yaw_adjustment = (delta_x / self.image_width) * self.HFOV
        pitch_adjustment = (delta_y / self.image_height) * self.VFOV
        return yaw_adjustment, pitch_adjustment

    def calculate_gimbal_offsets(self, direction):
        pitch_offset = -0.1
        if direction == "Moving Right":
            yaw_offset = 0.3
        elif direction == "Moving Left":
            yaw_offset = -0.3
        else:
            yaw_offset = 0
        return yaw_offset, pitch_offset
    
    def update_orientation(self, pitch, yaw):
        self.pitch = pitch
        self.yaw = yaw
        print(f'Updated Gimbal Orientation: Pitch={pitch}, Yaw={yaw}')
        
        
class GimbalOrientationSubscriber(Node):
    def __init__(self, gimbal):
        super().__init__('gimbal_orientation_subscriber')
        self.subscription = self.create_subscription(
            Float32MultiArray,
            'gimbal_angles',
            self.orientation_callback,
            10)
        self.subscription  # prevent unused variable warning
        self.gimbal = gimbal
        self.get_logger().info('Gimbal Subscriber Node has been started.')

    def orientation_callback(self, msg):
        pitch, yaw = msg.data
        self.gimbal.update_orientation(pitch, yaw)

