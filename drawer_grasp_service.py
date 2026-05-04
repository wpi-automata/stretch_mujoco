#!/usr/bin/env python3
"""
ROS2 DrawerGrasp service server.

Receives: RGB image + depth image + 3D bounding box of a detected drawer.
Returns: grasp point (handle location) for the robot to pull.

This is the "next step" callable service that processes drawer detections
from the explorer and computes optimal grasp points.
"""

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point

from stretch_drawer_interfaces.srv import DrawerGrasp


class DrawerGraspServer(Node):
    def __init__(self):
        super().__init__("drawer_grasp_server")
        self.srv = self.create_service(DrawerGrasp, "/drawer_grasp", self.handle_grasp_request)
        self.get_logger().info("DrawerGrasp service ready on /drawer_grasp")

    def handle_grasp_request(self, request, response):
        """
        Process the drawer detection and compute the optimal grasp point.
        Strategy: find the handle as the center of the front face, offset
        slightly toward the camera.
        """
        self.get_logger().info("Received DrawerGrasp request")

        # Unpack bounding box corners
        corners = np.array([[p.x, p.y, p.z] for p in request.bbox_corners])
        centroid = np.array([
            request.drawer_centroid.x,
            request.drawer_centroid.y,
            request.drawer_centroid.z,
        ])
        pull_dir = np.array([
            request.pull_direction.x,
            request.pull_direction.y,
            request.pull_direction.z,
        ])

        # Decode images for analysis
        rgb = self._decode_rgb(request.rgb_image)
        depth = self._decode_depth(request.depth_image)

        # Compute grasp point: handle is at vertical center of front face
        grasp_point = self._compute_grasp_point(corners, centroid, pull_dir, rgb, depth)

        if grasp_point is not None:
            response.grasp_point = Point(
                x=float(grasp_point[0]),
                y=float(grasp_point[1]),
                z=float(grasp_point[2]),
            )
            response.success = True
            response.message = "Grasp point computed successfully"
            self.get_logger().info(f"Grasp point: [{grasp_point[0]:.3f}, {grasp_point[1]:.3f}, {grasp_point[2]:.3f}]")
        else:
            response.success = False
            response.message = "Failed to compute grasp point"
            self.get_logger().warn("Could not determine grasp point")

        return response

    def _compute_grasp_point(self, corners, centroid, pull_dir, rgb, depth):
        """
        Compute optimal grasp point on the drawer handle.

        Strategy:
          1. The handle is typically at the vertical center of the front face
          2. Horizontally centered on the drawer
          3. Slightly protruding in the pull direction
        """
        # Front face: corners closest to camera (along pull direction)
        # Sort corners by projection onto pull_dir
        projections = corners @ pull_dir
        front_indices = np.argsort(projections)[-4:]  # 4 corners most in pull direction
        front_corners = corners[front_indices]

        # Handle is at the center of the front face, biased toward vertical middle
        handle_center = front_corners.mean(axis=0)

        # Offset slightly outward (handle protrudes from face)
        handle_offset = 0.015  # 1.5 cm
        grasp_point = handle_center + pull_dir * handle_offset

        # Refine using depth image if available
        if depth is not None and depth.size > 0:
            grasp_point = self._refine_with_depth(grasp_point, depth, centroid, pull_dir)

        return grasp_point

    def _refine_with_depth(self, initial_grasp, depth, centroid, pull_dir):
        """
        Refine grasp point using depth discontinuity (handle protrudes).
        Look for the closest point in the depth near the centroid region.
        """
        h, w = depth.shape[:2]
        center_region = depth[h//3:2*h//3, w//3:2*w//3]

        if center_region.size == 0:
            return initial_grasp

        # The handle is the closest (smallest depth) point in the center region
        valid = center_region[center_region > 0.01]
        if len(valid) == 0:
            return initial_grasp

        min_depth = valid.min()
        mean_depth = valid.mean()

        # If there's a significant protrusion (handle sticks out > 1cm)
        if mean_depth - min_depth > 0.01:
            protrusion = mean_depth - min_depth
            return initial_grasp + pull_dir * protrusion

        return initial_grasp

    @staticmethod
    def _decode_rgb(img_msg):
        """Decode sensor_msgs/Image to numpy array."""
        if not img_msg.data:
            return None
        arr = np.frombuffer(bytes(img_msg.data), dtype=np.uint8)
        return arr.reshape(img_msg.height, img_msg.width, 3)

    @staticmethod
    def _decode_depth(img_msg):
        """Decode depth sensor_msgs/Image to numpy array."""
        if not img_msg.data:
            return None
        arr = np.frombuffer(bytes(img_msg.data), dtype=np.float32)
        return arr.reshape(img_msg.height, img_msg.width)


def main():
    rclpy.init()
    node = DrawerGraspServer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
