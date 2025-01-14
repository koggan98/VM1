#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster
from scipy.spatial.transform import Rotation as R
import numpy as np


class DynamicTFPublisher(Node):
    def __init__(self):
        super().__init__('dynamic_tf_publisher')
        self.tf_broadcaster = TransformBroadcaster(self)

        # Subscriber for marker-to-base transformations (T_m_b)
        self.subscription_T_m_b = self.create_subscription(
            Float64MultiArray,
            'transformation_matrix',
            self.transformation_callback,
            10
        )

        # Subscriber for camera-to-marker transformations (T_c_m)
        self.subscription_T_c_m = self.create_subscription(
            Float64MultiArray,
            'camera_to_marker_matrix',
            self.transformation_callback,
            10
        )

        # Subscriber for the hand marker
        self.subscription_hand_marker = self.create_subscription(
            Marker,
            'hand_marker',
            self.hand_marker_callback,
            10
        )

        # Store transformations
        self.transformation_matrices_T_m_b = None
        self.transformation_matrices_T_c_m = None

        # Publisher for the hand marker visualization
        self.marker_publisher = self.create_publisher(Marker, 'visualized_hand_marker', 10)

    def transformation_callback(self, msg):
        # Determine which type of message is received based on the topic
        topic_name = msg._topic_name

        matrices = np.array(msg.data).reshape(-1, 4, 4)

        if topic_name == '/transformation_matrix':
            self.transformation_matrices_T_m_b = matrices
            self.get_logger().info("Received transformation matrices T_m_b")
        elif topic_name == '/camera_to_marker_matrix':
            self.transformation_matrices_T_c_m = matrices
            self.get_logger().info("Received transformation matrices T_c_m")

        self.publish_all_tfs()

    def publish_all_tfs(self):
        if self.transformation_matrices_T_m_b is not None and self.transformation_matrices_T_c_m is not None:
            num_markers = min(len(self.transformation_matrices_T_m_b), len(self.transformation_matrices_T_c_m))

            for i in range(num_markers):
                T_m_b = self.transformation_matrices_T_m_b[i]
                T_c_m = self.transformation_matrices_T_c_m[i]

                marker_frame = f'marker_{i}'
                camera_frame = f'camera_{i}'

                # Publish the marker-to-base transformation
                self.publish_tf(T_m_b, 'robot_base', marker_frame)

                # Publish the camera-to-marker transformation
                self.publish_tf(T_c_m, marker_frame, camera_frame)

    def hand_marker_callback(self, msg):
        # Publish the hand marker as a blue sphere
        marker = Marker()
        marker.header.frame_id = msg.header.frame_id
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD

        # Set the position of the marker
        marker.pose.position.x = msg.pose.position.x
        marker.pose.position.y = msg.pose.position.y
        marker.pose.position.z = msg.pose.position.z

        # Marker size
        marker.scale.x = 0.05
        marker.scale.y = 0.05
        marker.scale.z = 0.05

        # Marker color (blue)
        marker.color.r = 0.0
        marker.color.g = 0.0
        marker.color.b = 1.0
        marker.color.a = 1.0

        # Publish the marker
        self.marker_publisher.publish(marker)
        self.get_logger().info("Published hand marker as blue sphere")

    def publish_tf(self, matrix, parent_frame, child_frame):
        transform = TransformStamped()
        transform.header.stamp = self.get_clock().now().to_msg()
        transform.header.frame_id = parent_frame
        transform.child_frame_id = child_frame

        # Extract translation
        transform.transform.translation.x = matrix[0, 3]
        transform.transform.translation.y = matrix[1, 3]
        transform.transform.translation.z = matrix[2, 3]

        # Extract rotation as quaternion
        rotation_matrix = matrix[:3, :3]
        quat = R.from_matrix(rotation_matrix).as_quat()
        transform.transform.rotation.x = quat[0]
        transform.transform.rotation.y = quat[1]
        transform.transform.rotation.z = quat[2]
        transform.transform.rotation.w = quat[3]

        # Send the transform
        self.tf_broadcaster.sendTransform(transform)
        self.get_logger().info(f"Publishing TF: {parent_frame} -> {child_frame}")


def main(args=None):
    rclpy.init(args=args)
    node = DynamicTFPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
