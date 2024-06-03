import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

# 3 states: rotation, front_turret_focus, back_turret_focus
class FSM(Node):
    def __init__(self, namespace1, namespace2):
        super().__init__('fsm_node')
        self.state = "rotation"
        
        self.front_camera_target_information_subscription = self.create_subscription(
            Float32MultiArray,
            f'{namespace1}/target_information',
            self.front_target_information_callback,
            10)
        self.front_camera_target_information_subscription

        self.back_camera_target_information_subscription = self.create_subscription(
            Float32MultiArray,
            f'{namespace2}/target_information',
            self.back_target_information_callback,
            10)
        self.back_camera_target_information_subscription

        # decision contains the pitch and yaw to turn the gimbal by, taking into account all 3 states
        self.decision_publisher = self.create_publisher(Float32MultiArray, 'decision', 10)
        self.timer = self.create_timer(0.03, self.publish_decision)

        self.front_euclidean_dist, self.front_pitch, self.front_yaw = None, None, None
        self.back_euclidean_dist, self.back_pitch, self.back_yaw = None, None, None
        
    def front_target_information_callback(self, msg):
        self.front_euclidean_dist, self.front_pitch, self.front_yaw = msg.data
        
    def back_target_information_callback(self, msg):
        self.back_euclidean_dist, self.back_pitch, self.back_yaw = msg.data
        
    def publish_decision(self):
        # rclpy.spin_once(self, timeout_sec=0.02)
        self.compute_state(self.front_euclidean_dist, self.back_euclidean_dist)

        pitch_to_turn, yaw_to_turn = 0, 0
        if self.state == "rotation":
            pitch_to_turn = 0
            yaw_to_turn = 0.5
        elif self.state == "front_turret_focus":
            pitch_to_turn = self.front_pitch
            yaw_to_turn = self.front_yaw
        elif self.state == "back_turret_focus":
            pitch_to_turn = -self.back_pitch
            yaw_to_turn = -self.back_yaw
        
        decision = Float32MultiArray()
        decision.data = [pitch_to_turn, yaw_to_turn]
        self.decision_publisher.publish(decision)

    def compute_state(self, front_camera_dist=None, back_camera_dist=None):
        if front_camera_dist is not None and back_camera_dist is None:
            self.state = "front_turret_focus"
        elif back_camera_dist is not None and front_camera_dist is None:
            self.state = "back_turret_focus"
        elif front_camera_dist is not None and back_camera_dist is not None:
            if front_camera_dist < back_camera_dist:
                self.state = "front_turret_focus"
            else:
                self.state = "back_turret_focus"
        else:
            self.state = "rotation"
    
    def get_state(self):
        return self.state
