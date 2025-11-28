import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import time

class TrajPublisher(Node):
    def __init__(self):
        super().__init__("traj_pub_loop")

        self.pub = self.create_publisher(
            JointTrajectory,
            "/arm_controller/joint_trajectory",
            10
        )

        self.timer = self.create_timer(0.1, self.timer_callback)  # 10 Hz

        self.start_time = self.get_clock().now()

        # Commande que tu veux envoyer
        self.positions = [0.5, -0.3, 0.2, 0.0, 0.0]

    def timer_callback(self):

        msg = JointTrajectory()
        msg.joint_names = ['R0_Yaw','R1_Pitch','R2_Pitch','R3_Yaw','R4_Pitch']

        point = JointTrajectoryPoint()
        point.positions = self.positions
        point.time_from_start.sec = 1
        point.time_from_start.nanosec = 0

        msg.points.append(point)

        self.pub.publish(msg)
        self.get_logger().info("Trajectory sent (loop)")

def main():
    rclpy.init()
    node = TrajPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
