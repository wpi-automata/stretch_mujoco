#!/usr/bin/env python3
"""
Stretch3 Drawer Explorer: random walk + SAM segmentation + grasp & pull.

Behavior:
  1. Random walk via mobile base while panning head camera
  2. Use SAM to segment drawers from wrist (d405) and head (d435i) cameras
  3. Compute 3D bounding box of detected drawer from depth
  4. Call ROS2 DrawerGrasp service with image + bbox (next-step planner)
  5. Navigate to drawer, find handle, grasp, and pull until force limit
"""

import sys
import os
import time
import threading

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, "third_party", "robosuite"))
sys.path.insert(0, os.path.join(_here, "third_party", "robocasa"))

import numpy as np
import robosuite
from robosuite.utils.camera_utils import (
    get_camera_extrinsic_matrix,
    get_camera_intrinsic_matrix,
    get_real_depth_map,
)

import robocasa  # noqa: F401 — registers tasks

# Optional ROS2 imports (script works standalone without ROS)
try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import Image
    from geometry_msgs.msg import Point, Vector3, Transform, Quaternion
    from visualization_msgs.msg import Marker, MarkerArray
    from std_msgs.msg import Header, ColorRGBA
    HAS_ROS2 = True
except ImportError:
    HAS_ROS2 = False

# SAM imports
try:
    from segment_anything import sam_model_registry, SamPredictor
    HAS_SAM = True
except ImportError:
    try:
        from mobile_sam import sam_model_registry, SamPredictor
        HAS_SAM = True
    except ImportError:
        HAS_SAM = False

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

IMG_HEIGHT = 480
IMG_WIDTH = 640
HEAD_CAM = "robot0_d435i_camera_rgb"
WRIST_CAM = "robot0_d405_rgb"

SAM_CHECKPOINT = os.environ.get("SAM_CHECKPOINT", os.path.expanduser("~/sam_vit_h_4b8939.pth"))
SAM_MODEL_TYPE = os.environ.get("SAM_MODEL_TYPE", "vit_h")

DRAWER_PROMPT = "drawer"

# Stretch3 workspace limits (from XML joint ranges and kinematic chain)
#
# IMPORTANT: The Stretch3 arm extends PERPENDICULAR to the drive direction.
# The arm is mounted on the left side of the mast and telescopes outward
# to the robot's left. The robot must position itself beside the target,
# not in front of it.
#
# Kinematic chain offsets (from XML):
#   base_link → link_lift: pos=(-0.104, +0.135, +0.2) — mast is left of center
#   link_arm_l4 geoms at (-0.2547, 0, 0) in lift frame
#   After quat rotation, arm extends roughly in -X (robot's left) direction
#
# Lift: 0.0–1.1 m vertical travel
LIFT_MIN = 0.0
LIFT_MAX = 1.1
BASE_FLOOR_OFFSET = 0.2  # mast base height above floor
EEF_HEIGHT_MIN = BASE_FLOOR_OFFSET + LIFT_MIN  # ~0.2 m
EEF_HEIGHT_MAX = BASE_FLOOR_OFFSET + LIFT_MAX  # ~1.3 m

# Arm extension: 4 telescope segments × 0.13 m = 0.52 m max
ARM_EXTENSION_MAX = 4 * 0.13  # 0.52 m
ARM_EXTENSION_MIN = 0.05

# The fixed arm structure (link_arm_l4) provides ~0.27 m of reach before
# the telescoping segments begin. Shoulder mounts ~0.14 m from base center.
ARM_FIXED_LINK_LENGTH = 0.27  # link_arm_l4 offset
SHOULDER_LATERAL_OFFSET = 0.135  # mast Y offset from base center

# Total lateral reach from base center (perpendicular to drive direction)
MAX_LATERAL_REACH = SHOULDER_LATERAL_OFFSET + ARM_FIXED_LINK_LENGTH + ARM_EXTENSION_MAX  # ~0.92 m
MIN_LATERAL_REACH = SHOULDER_LATERAL_OFFSET + ARM_FIXED_LINK_LENGTH + ARM_EXTENSION_MIN  # ~0.45 m

# Along the drive direction, the arm has very limited reach (only wrist flexibility)
MAX_FORWARD_REACH_AT_TARGET = 0.15  # how far forward/back from arm-plane the EEF can deviate

# Force threshold for pull termination (Nm)
PULL_FORCE_THRESHOLD = 15.0
MAX_PULL_DISTANCE = 0.4  # meters

STRETCH3_CONTROLLER_CONFIG = {
    "type": "BASIC",
    "body_parts": {
        "right": {
            "type": "JOINT_POSITION",
            "input_max": 1,
            "input_min": -1,
            "output_max": 0.5,
            "output_min": -0.5,
            "kp": 300,
            "kd": 50,
            "kv": 200,
            "velocity_limits": [-1, 1],
            "kp_limits": [0, 1000],
            "interpolation": None,
            "ramp_ratio": 0.2,
            "gripper": {
                "type": "GRIP",
            },
        },
        "base": {
            "type": "JOINT_VELOCITY",
            "interpolation": None,
        },
        "head": {
            "type": "JOINT_POSITION",
            "input_max": 1,
            "input_min": -1,
            "output_max": 0.5,
            "output_min": -0.5,
            "kp": 300,
            "kd": 50,
            "kv": 200,
            "velocity_limits": [-1, 1],
            "kp_limits": [0, 1000],
            "interpolation": None,
            "ramp_ratio": 0.2,
        },
    },
}


# ---------------------------------------------------------------------------
# SAM Drawer Detector
# ---------------------------------------------------------------------------

class DrawerDetector:
    """Uses SAM to segment drawer regions from RGB images."""

    def __init__(self):
        self.predictor = None
        if HAS_SAM and os.path.exists(SAM_CHECKPOINT):
            print(f"[DrawerDetector] Loading SAM model: {SAM_MODEL_TYPE}")
            sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_CHECKPOINT)
            sam.to("cuda" if self._has_cuda() else "cpu")
            self.predictor = SamPredictor(sam)
            print("[DrawerDetector] SAM loaded.")
        else:
            print("[DrawerDetector] SAM not available — using fallback edge detector.")

    @staticmethod
    def _has_cuda():
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def detect_drawers(self, rgb_image):
        """
        Detect drawer regions in the image.

        Returns list of dicts: [{"mask": np.array, "bbox": (x1, y1, x2, y2), "score": float}]
        """
        if self.predictor is not None:
            return self._detect_with_sam(rgb_image)
        return self._detect_fallback(rgb_image)

    def _detect_with_sam(self, rgb_image):
        """Use SAM with a grid of point prompts, then filter for rectangular drawer-like masks."""
        self.predictor.set_image(rgb_image)

        h, w = rgb_image.shape[:2]
        # Grid prompt: sample points across the image
        grid_points = []
        for yi in range(3, h - 3, h // 5):
            for xi in range(3, w - 3, w // 5):
                grid_points.append([xi, yi])
        grid_points = np.array(grid_points)
        grid_labels = np.ones(len(grid_points), dtype=int)

        masks, scores, _ = self.predictor.predict(
            point_coords=grid_points,
            point_labels=grid_labels,
            multimask_output=True,
        )

        detections = []
        for i, (mask, score) in enumerate(zip(masks, scores)):
            if score < 0.5:
                continue
            bbox = self._mask_to_bbox(mask)
            if bbox is None:
                continue
            if self._is_drawer_like(mask, bbox):
                detections.append({"mask": mask, "bbox": bbox, "score": float(score)})

        # Sort by score, return top detections
        detections.sort(key=lambda d: d["score"], reverse=True)
        return detections[:5]

    def _detect_fallback(self, rgb_image):
        """Simple edge-based drawer detection when SAM isn't available."""
        gray = np.mean(rgb_image, axis=2).astype(np.uint8)

        # Sobel edge detection
        dx = np.abs(np.diff(gray.astype(float), axis=1))
        dy = np.abs(np.diff(gray.astype(float), axis=0))

        edges = np.zeros_like(gray, dtype=float)
        edges[:, :-1] += dx
        edges[:-1, :] += dy
        edges = (edges > 30).astype(np.uint8)

        # Find rectangular contour regions (simplified)
        h, w = gray.shape
        detections = []

        # Sliding window search for rectangular high-edge-density regions
        for scale in [0.15, 0.25, 0.35]:
            bh, bw = int(h * scale * 0.4), int(w * scale)
            stride = max(bh // 2, 1)
            for y in range(0, h - bh, stride):
                for x in range(0, w - bw, stride):
                    patch = edges[y:y+bh, x:x+bw]
                    edge_density = patch.sum() / (bh * bw)
                    # Drawers have edges at boundaries
                    border_density = (
                        patch[0, :].sum() + patch[-1, :].sum() +
                        patch[:, 0].sum() + patch[:, -1].sum()
                    ) / (2 * (bh + bw))

                    if edge_density > 0.05 and border_density > 0.2:
                        mask = np.zeros((h, w), dtype=bool)
                        mask[y:y+bh, x:x+bw] = True
                        detections.append({
                            "mask": mask,
                            "bbox": (x, y, x + bw, y + bh),
                            "score": float(border_density),
                        })

        # Non-max suppression (simple IoU)
        detections.sort(key=lambda d: d["score"], reverse=True)
        kept = []
        for det in detections:
            overlap = False
            for k in kept:
                if self._iou(det["bbox"], k["bbox"]) > 0.3:
                    overlap = True
                    break
            if not overlap:
                kept.append(det)
        return kept[:5]

    @staticmethod
    def _mask_to_bbox(mask):
        ys, xs = np.where(mask)
        if len(ys) == 0:
            return None
        return (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))

    @staticmethod
    def _is_drawer_like(mask, bbox):
        """Check if mask is roughly rectangular and has drawer-like aspect ratio."""
        x1, y1, x2, y2 = bbox
        bw, bh = x2 - x1, y2 - y1
        if bw < 20 or bh < 10:
            return False
        aspect = bw / max(bh, 1)
        # Drawers are typically wider than tall
        if aspect < 0.8 or aspect > 5.0:
            return False
        # Check mask fills most of bbox (rectangular)
        bbox_area = bw * bh
        mask_area = mask.sum()
        fill_ratio = mask_area / max(bbox_area, 1)
        return fill_ratio > 0.5

    @staticmethod
    def _iou(bbox1, bbox2):
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        return inter / max(area1 + area2 - inter, 1)


# ---------------------------------------------------------------------------
# 3D Bounding Box Computation
# ---------------------------------------------------------------------------

def compute_3d_bbox(env, depth_obs, mask, camera_name):
    """
    Given a depth image and a 2D mask, back-project masked pixels to 3D
    and return oriented bounding box corners + centroid + pull direction.
    """
    sim = env.sim
    depth_2d = depth_obs.squeeze() if depth_obs.ndim == 3 else depth_obs
    real_depth = get_real_depth_map(sim, depth_2d)

    # Get camera transforms
    cam_extrinsic = get_camera_extrinsic_matrix(sim, camera_name)
    cam_intrinsic = get_camera_intrinsic_matrix(sim, camera_name, IMG_HEIGHT, IMG_WIDTH)

    # Get masked pixel coordinates
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return None

    # Subsample if too many points
    if len(ys) > 1000:
        idx = np.random.choice(len(ys), 1000, replace=False)
        ys, xs = ys[idx], xs[idx]

    # Back-project manually: pixel (u,v) + depth -> 3D camera frame -> world frame
    fx, fy = cam_intrinsic[0, 0], cam_intrinsic[1, 1]
    cx, cy = cam_intrinsic[0, 2], cam_intrinsic[1, 2]

    depths = real_depth[ys, xs]
    valid_depth = (depths > 0.05) & (depths < 3.0)
    ys, xs, depths = ys[valid_depth], xs[valid_depth], depths[valid_depth]

    if len(depths) < 10:
        return None

    # Camera frame coordinates
    x_cam = (xs - cx) * depths / fx
    y_cam = (ys - cy) * depths / fy
    z_cam = depths

    # Stack as (N, 4) homogeneous
    pts_cam = np.stack([x_cam, y_cam, z_cam, np.ones_like(z_cam)], axis=-1)

    # Transform to world frame
    points_3d = (cam_extrinsic @ pts_cam.T).T[:, :3]

    # Filter outliers
    valid = (points_3d[:, 2] > 0.0) & (points_3d[:, 2] < 3.0)
    points_3d = points_3d[valid]

    if len(points_3d) < 10:
        return None

    # Axis-aligned bounding box in world frame
    mins = points_3d.min(axis=0)
    maxs = points_3d.max(axis=0)
    centroid = (mins + maxs) / 2.0

    # 8 corners of the AABB
    corners = np.array([
        [mins[0], mins[1], mins[2]],
        [maxs[0], mins[1], mins[2]],
        [maxs[0], maxs[1], mins[2]],
        [mins[0], maxs[1], mins[2]],
        [mins[0], mins[1], maxs[2]],
        [maxs[0], mins[1], maxs[2]],
        [maxs[0], maxs[1], maxs[2]],
        [mins[0], maxs[1], maxs[2]],
    ])

    # Pull direction: from drawer centroid toward camera (outward normal)
    cam_pos = cam_extrinsic[:3, 3]
    pull_dir = cam_pos - centroid
    pull_dir[2] = 0  # keep horizontal
    pull_dir = pull_dir / (np.linalg.norm(pull_dir) + 1e-8)

    return {
        "corners": corners,
        "centroid": centroid,
        "pull_direction": pull_dir,
        "points_3d": points_3d,
        "cam_extrinsic": cam_extrinsic,
    }


def is_drawer_reachable(bbox_result, robot_base_pos, robot_base_quat=None):
    """
    Check if the drawer centroid is within the Stretch3's reachable workspace.

    The Stretch3 arm extends LATERALLY (perpendicular to the drive direction,
    to the robot's left). The workspace is NOT a sphere around the base — it's
    a vertical slice to the robot's left side.

    Workspace constraints:
      - Height: EEF_HEIGHT_MIN to EEF_HEIGHT_MAX (lift range)
      - Lateral distance (perpendicular to forward): MIN_LATERAL_REACH to MAX_LATERAL_REACH
      - Forward distance (along drive axis): within MAX_FORWARD_REACH_AT_TARGET

    If robot_base_quat is None, we only check height and total horizontal distance
    (the robot can always rotate to face the right direction).

    Returns (reachable: bool, reason: str)
    """
    centroid = bbox_result["centroid"]

    # 1. Height check: drawer handle must be within lift range
    drawer_height = centroid[2]
    if drawer_height < EEF_HEIGHT_MIN:
        return False, f"too low ({drawer_height:.2f}m < {EEF_HEIGHT_MIN:.2f}m)"
    if drawer_height > EEF_HEIGHT_MAX:
        return False, f"too high ({drawer_height:.2f}m > {EEF_HEIGHT_MAX:.2f}m)"

    # 2. Horizontal distance check
    dx = centroid[0] - robot_base_pos[0]
    dy = centroid[1] - robot_base_pos[1]
    horizontal_dist = np.sqrt(dx**2 + dy**2)

    # If we don't know the robot's heading, check if the drawer is
    # within reach assuming the robot can rotate to optimal position
    # (the robot must be positioned so the drawer is to its left at
    # the right lateral distance)
    if horizontal_dist > MAX_LATERAL_REACH:
        return False, f"too far ({horizontal_dist:.2f}m > {MAX_LATERAL_REACH:.2f}m max lateral reach)"
    if horizontal_dist < MIN_LATERAL_REACH:
        return False, f"too close ({horizontal_dist:.2f}m < {MIN_LATERAL_REACH:.2f}m min lateral reach)"

    # 3. If we have orientation, check that the drawer is actually to the
    #    robot's left (arm side) and within the forward tolerance
    if robot_base_quat is not None:
        # Extract robot yaw from quaternion (assuming quat = [w, x, y, z])
        if len(robot_base_quat) == 4:
            w, x, y, z = robot_base_quat[0], robot_base_quat[1], robot_base_quat[2], robot_base_quat[3]
            robot_yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y**2 + z**2))
        else:
            robot_yaw = 0.0

        # Robot's forward direction
        fwd = np.array([np.cos(robot_yaw), np.sin(robot_yaw)])
        # Robot's left direction (arm side)
        left = np.array([-np.sin(robot_yaw), np.cos(robot_yaw)])

        # Project drawer offset onto robot's frame
        offset = np.array([dx, dy])
        lateral_component = np.dot(offset, left)   # positive = to robot's left (arm side)
        forward_component = np.dot(offset, fwd)    # positive = in front of robot

        if lateral_component < 0:
            return False, f"drawer is to robot's RIGHT (lateral={lateral_component:.2f}m), arm is on LEFT"

        if lateral_component < MIN_LATERAL_REACH:
            return False, f"too close laterally ({lateral_component:.2f}m < {MIN_LATERAL_REACH:.2f}m)"
        if lateral_component > MAX_LATERAL_REACH:
            return False, f"too far laterally ({lateral_component:.2f}m > {MAX_LATERAL_REACH:.2f}m)"

        if abs(forward_component) > MAX_FORWARD_REACH_AT_TARGET:
            return False, f"too far along drive axis (|{forward_component:.2f}|m > {MAX_FORWARD_REACH_AT_TARGET:.2f}m)"

    return True, "reachable"


def compute_approach_pose(drawer_centroid, robot_base_pos):
    """
    Compute where the robot should position itself to reach the drawer.

    The Stretch3 must park itself so the drawer is to its LEFT at the
    optimal lateral distance. The robot's forward axis should be parallel
    to the drawer face (perpendicular to the pull direction).

    Returns (target_pos_2d, target_yaw) — the robot's base position and
    heading it needs to achieve for the arm to reach the drawer.
    """
    dx = drawer_centroid[0] - robot_base_pos[0]
    dy = drawer_centroid[1] - robot_base_pos[1]

    # Angle from robot to drawer
    angle_to_drawer = np.arctan2(dy, dx)

    # The arm extends to the robot's left. So the robot's forward direction
    # should be 90° clockwise from the direction to the drawer.
    # (If drawer is to robot's left, forward is perpendicular to that.)
    target_yaw = angle_to_drawer - np.pi / 2.0

    # Optimal lateral distance: middle of reach range
    optimal_lateral = (MIN_LATERAL_REACH + MAX_LATERAL_REACH) / 2.0

    # Target position: offset from drawer centroid so that the drawer
    # ends up at optimal_lateral to the robot's left
    # Robot's left at target_yaw is: (-sin(target_yaw), cos(target_yaw))
    # Drawer should be at robot_pos + optimal_lateral * left_dir
    # So robot_pos = drawer_pos - optimal_lateral * left_dir
    left_dir = np.array([-np.sin(target_yaw), np.cos(target_yaw)])
    target_pos_2d = drawer_centroid[:2] - optimal_lateral * left_dir

    return target_pos_2d, target_yaw


def find_handle_grasp_point(bbox_result):
    """
    Estimate the handle position on the drawer front face.
    The handle is typically at the vertical center, horizontally centered,
    slightly in front of the drawer face.
    """
    centroid = bbox_result["centroid"]
    pull_dir = bbox_result["pull_direction"]

    # Handle is slightly in front of the centroid (toward the camera)
    grasp_point = centroid + pull_dir * 0.02
    return grasp_point


# ---------------------------------------------------------------------------
# ROS2 Service Client (calls DrawerGrasp service)
# ---------------------------------------------------------------------------

class DrawerGraspClient:
    """ROS2 service client that sends drawer detection to the grasp planner."""

    def __init__(self):
        self.node = None
        self.client = None
        if HAS_ROS2:
            rclpy.init()
            self.node = rclpy.create_node("drawer_explorer_client")
            # Import the service type — requires stretch_drawer_interfaces built
            try:
                from stretch_drawer_interfaces.srv import DrawerGrasp
                self.client = self.node.create_client(DrawerGrasp, "/drawer_grasp")
                self.srv_type = DrawerGrasp
                print("[ROS2] DrawerGrasp service client ready.")
            except ImportError:
                print("[ROS2] stretch_drawer_interfaces not built — service calls disabled.")
                self.client = None

    def call_service(self, rgb_image, depth_image, bbox_result):
        """Send drawer detection to grasp planning service."""
        if self.client is None:
            print("[ROS2] Service not available, skipping call.")
            return None

        if not self.client.wait_for_service(timeout_sec=2.0):
            print("[ROS2] DrawerGrasp service not available.")
            return None

        req = self.srv_type.Request()

        # Pack RGB image
        req.rgb_image = Image()
        req.rgb_image.height = rgb_image.shape[0]
        req.rgb_image.width = rgb_image.shape[1]
        req.rgb_image.encoding = "rgb8"
        req.rgb_image.step = rgb_image.shape[1] * 3
        req.rgb_image.data = rgb_image.tobytes()

        # Pack depth image
        req.depth_image = Image()
        req.depth_image.height = depth_image.shape[0]
        req.depth_image.width = depth_image.shape[1]
        req.depth_image.encoding = "32FC1"
        req.depth_image.step = depth_image.shape[1] * 4
        req.depth_image.data = depth_image.astype(np.float32).tobytes()

        # Pack bounding box corners
        for i, corner in enumerate(bbox_result["corners"]):
            req.bbox_corners[i] = Point(x=float(corner[0]), y=float(corner[1]), z=float(corner[2]))

        req.drawer_centroid = Point(
            x=float(bbox_result["centroid"][0]),
            y=float(bbox_result["centroid"][1]),
            z=float(bbox_result["centroid"][2]),
        )

        req.pull_direction = Vector3(
            x=float(bbox_result["pull_direction"][0]),
            y=float(bbox_result["pull_direction"][1]),
            z=float(bbox_result["pull_direction"][2]),
        )

        # Camera extrinsic as Transform
        cam_ext = bbox_result["cam_extrinsic"]
        req.camera_to_world = Transform()
        req.camera_to_world.translation.x = float(cam_ext[0, 3])
        req.camera_to_world.translation.y = float(cam_ext[1, 3])
        req.camera_to_world.translation.z = float(cam_ext[2, 3])
        req.camera_to_world.rotation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)

        future = self.client.call_async(req)
        rclpy.spin_until_future_complete(self.node, future, timeout_sec=10.0)

        if future.result() is not None:
            resp = future.result()
            print(f"[ROS2] Grasp service response: success={resp.success}, msg={resp.message}")
            if resp.success:
                return np.array([resp.grasp_point.x, resp.grasp_point.y, resp.grasp_point.z])
        return None

    def shutdown(self):
        if self.node:
            self.node.destroy_node()
            rclpy.shutdown()


# ---------------------------------------------------------------------------
# ROS2 Visualization Publisher
# ---------------------------------------------------------------------------

class RVizPublisher:
    """Publishes camera images, drawer bounding boxes, and robot state to ROS2 for RViz."""

    def __init__(self):
        self.node = None
        if not HAS_ROS2:
            print("[RViz] ROS2 not available — visualization disabled.")
            return

        if not rclpy.ok():
            rclpy.init()

        self.node = rclpy.create_node("drawer_explorer_viz")
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)

        # Camera image publishers
        self.head_rgb_pub = self.node.create_publisher(Image, "/drawer_explorer/head_camera/image_raw", qos)
        self.wrist_rgb_pub = self.node.create_publisher(Image, "/drawer_explorer/wrist_camera/image_raw", qos)
        self.head_depth_pub = self.node.create_publisher(Image, "/drawer_explorer/head_camera/depth", qos)
        self.wrist_depth_pub = self.node.create_publisher(Image, "/drawer_explorer/wrist_camera/depth", qos)

        # Bounding box marker publisher
        self.marker_pub = self.node.create_publisher(MarkerArray, "/drawer_explorer/drawer_bbox", 10)

        # Detection image with drawn bbox
        self.detection_pub = self.node.create_publisher(Image, "/drawer_explorer/detection_image", qos)

        self._marker_id = 0
        print("[RViz] Visualization publishers ready.")

    def publish_cameras(self, obs):
        """Publish camera observations as ROS Image messages."""
        if self.node is None:
            return

        stamp = self.node.get_clock().now().to_msg()

        # Head camera RGB
        head_rgb_key = f"{HEAD_CAM}_image"
        if head_rgb_key in obs:
            self.head_rgb_pub.publish(self._numpy_to_image_msg(obs[head_rgb_key], "rgb8", "head_camera", stamp))

        # Wrist camera RGB
        wrist_rgb_key = f"{WRIST_CAM}_image"
        if wrist_rgb_key in obs:
            self.wrist_rgb_pub.publish(self._numpy_to_image_msg(obs[wrist_rgb_key], "rgb8", "wrist_camera", stamp))

        # Head depth
        head_depth_key = f"{HEAD_CAM}_depth"
        if head_depth_key in obs:
            self.head_depth_pub.publish(self._numpy_to_image_msg(obs[head_depth_key], "32FC1", "head_camera", stamp))

        # Wrist depth
        wrist_depth_key = f"{WRIST_CAM}_depth"
        if wrist_depth_key in obs:
            self.wrist_depth_pub.publish(self._numpy_to_image_msg(obs[wrist_depth_key], "32FC1", "wrist_camera", stamp))

    def publish_drawer_bbox(self, bbox_result, reachable=True):
        """Publish the 3D bounding box of a detected drawer as a MarkerArray."""
        if self.node is None or bbox_result is None:
            return

        stamp = self.node.get_clock().now().to_msg()
        markers = MarkerArray()

        # Line strip for the bounding box wireframe
        bbox_marker = Marker()
        bbox_marker.header = Header(stamp=stamp, frame_id="world")
        bbox_marker.ns = "drawer_bbox"
        bbox_marker.id = self._marker_id
        bbox_marker.type = Marker.LINE_LIST
        bbox_marker.action = Marker.ADD
        bbox_marker.scale.x = 0.005  # line width

        # Green if reachable, red if not
        color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=0.9) if reachable else ColorRGBA(r=1.0, g=0.0, b=0.0, a=0.9)
        bbox_marker.color = color

        corners = bbox_result["corners"]
        # 12 edges of the AABB
        edges = [
            (0,1),(1,2),(2,3),(3,0),  # bottom face
            (4,5),(5,6),(6,7),(7,4),  # top face
            (0,4),(1,5),(2,6),(3,7),  # vertical edges
        ]
        for i, j in edges:
            p1 = Point(x=float(corners[i][0]), y=float(corners[i][1]), z=float(corners[i][2]))
            p2 = Point(x=float(corners[j][0]), y=float(corners[j][1]), z=float(corners[j][2]))
            bbox_marker.points.append(p1)
            bbox_marker.points.append(p2)

        markers.markers.append(bbox_marker)

        # Centroid sphere
        centroid_marker = Marker()
        centroid_marker.header = Header(stamp=stamp, frame_id="world")
        centroid_marker.ns = "drawer_centroid"
        centroid_marker.id = self._marker_id
        centroid_marker.type = Marker.SPHERE
        centroid_marker.action = Marker.ADD
        centroid_marker.pose.position = Point(
            x=float(bbox_result["centroid"][0]),
            y=float(bbox_result["centroid"][1]),
            z=float(bbox_result["centroid"][2]),
        )
        centroid_marker.scale.x = 0.03
        centroid_marker.scale.y = 0.03
        centroid_marker.scale.z = 0.03
        centroid_marker.color = ColorRGBA(r=1.0, g=1.0, b=0.0, a=1.0)
        markers.markers.append(centroid_marker)

        # Pull direction arrow
        arrow_marker = Marker()
        arrow_marker.header = Header(stamp=stamp, frame_id="world")
        arrow_marker.ns = "pull_direction"
        arrow_marker.id = self._marker_id
        arrow_marker.type = Marker.ARROW
        arrow_marker.action = Marker.ADD
        start = bbox_result["centroid"]
        end = start + bbox_result["pull_direction"] * 0.2
        arrow_marker.points.append(Point(x=float(start[0]), y=float(start[1]), z=float(start[2])))
        arrow_marker.points.append(Point(x=float(end[0]), y=float(end[1]), z=float(end[2])))
        arrow_marker.scale.x = 0.01  # shaft diameter
        arrow_marker.scale.y = 0.02  # head diameter
        arrow_marker.color = ColorRGBA(r=0.0, g=0.5, b=1.0, a=1.0)
        markers.markers.append(arrow_marker)

        self.marker_pub.publish(markers)
        self._marker_id += 1

    def publish_detection_image(self, rgb, bbox_2d, reachable=True):
        """Publish the camera image with the 2D bounding box drawn on it."""
        if self.node is None:
            return

        # Draw bbox on image copy
        img = rgb.copy()
        if img.dtype != np.uint8:
            img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)

        x1, y1, x2, y2 = bbox_2d
        color = (0, 255, 0) if reachable else (255, 0, 0)  # green or red
        # Draw rectangle (no opencv dependency — do it with numpy)
        thickness = 3
        img[y1:y1+thickness, x1:x2] = color
        img[y2-thickness:y2, x1:x2] = color
        img[y1:y2, x1:x1+thickness] = color
        img[y1:y2, x2-thickness:x2] = color

        stamp = self.node.get_clock().now().to_msg()
        self.detection_pub.publish(self._numpy_to_image_msg(img, "rgb8", "head_camera", stamp))

    def spin_once(self):
        """Process pending ROS callbacks (non-blocking)."""
        if self.node is not None:
            rclpy.spin_once(self.node, timeout_sec=0.0)

    def shutdown(self):
        if self.node:
            self.node.destroy_node()

    @staticmethod
    def _numpy_to_image_msg(arr, encoding, frame_id, stamp):
        msg = Image()
        msg.header = Header(stamp=stamp, frame_id=frame_id)
        if arr.ndim == 3:
            msg.height, msg.width = arr.shape[0], arr.shape[1]
            channels = arr.shape[2]
        else:
            msg.height, msg.width = arr.shape[0], arr.shape[1]
            channels = 1

        msg.encoding = encoding
        if encoding == "32FC1":
            data = arr.astype(np.float32)
            msg.step = msg.width * 4
        else:
            data = arr.astype(np.uint8) if arr.dtype != np.uint8 else arr
            msg.step = msg.width * channels
        msg.data = data.tobytes()
        return msg


# ---------------------------------------------------------------------------
# Motion Primitives
# ---------------------------------------------------------------------------

class Stretch3Controller:
    """High-level motion primitives for the Stretch3 in robosuite."""

    def __init__(self, env):
        self.env = env
        self.action_dim = env.action_spec[0].shape[0]
        # Action layout for Stretch3 with BASIC controller:
        #   [arm_joints (5: lift+4tele mapped to 1 coupled), wrist(3)] = right (8 action dims after coupling?)
        # Actually with our JOINT_POSITION controller the action is per-actuator:
        #   right: lift, arm, wrist_yaw, wrist_pitch, wrist_roll (5 actuators) + gripper(1)
        #   base: forward, unused_side, yaw (3) — differential drive, side must be 0
        #   head: pan, tilt, nav_cam (3)
        # Total: 5 + 1 + 3 + 3 = 12
        # But action spec includes all — let's figure it out from the env
        self._determine_action_layout()

    def _determine_action_layout(self):
        """Figure out which action indices control which body part."""
        # With BASIC composite controller, action is concatenated:
        # [right_arm_actions, gripper_action, base_actions, head_actions]
        robot = self.env.robots[0]
        # Get action dimensions from the composite controller
        self.n_actions = self.action_dim
        print(f"[Controller] Total action dim: {self.n_actions}")

        # Action layout from env (14 dims):
        # right: 8 joints (lift, arm_l3, arm_l2, arm_l1, arm_l0, wrist_yaw, wrist_pitch, wrist_roll)
        # gripper: 1
        # base: 3 (forward, side, yaw) — but Stretch3 is differential drive,
        #        so we only use forward (idx 9) and yaw (idx 11). Side (idx 10) must stay 0.
        # head: 2 (pan, tilt)
        # Total: 8 + 1 + 3 + 2 = 14
        self.right_slice = slice(0, 8)
        self.gripper_idx = 8
        self.base_fwd_idx = 9
        self.base_side_idx = 10  # NEVER use — Stretch3 cannot strafe
        self.base_yaw_idx = 11
        self.base_slice = slice(9, 12)
        self.head_slice = slice(12, 14)

    def zero_action(self):
        return np.zeros(self.n_actions)

    def random_walk_action(self):
        """Generate random base velocities for exploration (differential drive: forward + yaw only)."""
        action = self.zero_action()
        # Stretch3 is differential drive — forward/backward + rotate only, no lateral motion
        action[self.base_fwd_idx] = np.random.uniform(0.1, 0.5)
        action[self.base_yaw_idx] = np.random.uniform(-0.4, 0.4)
        return action

    def pan_head_action(self, pan_target, tilt_target):
        """Set head pan and tilt targets."""
        action = self.zero_action()
        action[12] = np.clip(pan_target, -1.0, 1.0)
        action[13] = np.clip(tilt_target, -1.0, 1.0)
        return action

    def move_to_grasp_action(self, current_eef_pos, target_pos):
        """
        Compute joint commands to move EEF toward target.

        The Stretch3 reaches laterally via telescope arm extension.
        Action layout (right_slice indices 0-7):
          0: lift (vertical)
          1-4: arm_l3, arm_l2, arm_l1, arm_l0 (telescope extension — lateral reach)
          5: wrist_yaw
          6: wrist_pitch
          7: wrist_roll
        """
        action = self.zero_action()
        delta = target_pos - current_eef_pos

        # Lift (index 0): adjust height to match drawer
        action[0] = np.clip(delta[2] * 2.0, -0.5, 0.5)

        # Arm telescope extension (indices 1-4): extend laterally toward target
        # All 4 segments extend together (coupled via tendon in real robot)
        horizontal_dist = np.sqrt(delta[0]**2 + delta[1]**2)
        extend_cmd = np.clip(horizontal_dist * 1.5, -0.3, 0.3)
        action[1] = extend_cmd  # arm_l3
        action[2] = extend_cmd  # arm_l2
        action[3] = extend_cmd  # arm_l1
        action[4] = extend_cmd  # arm_l0

        # Wrist pitch: tilt down slightly to align with drawer handle
        action[6] = np.clip(delta[2] * -0.5, -0.2, 0.2)

        return action

    def close_gripper_action(self):
        action = self.zero_action()
        action[self.gripper_idx] = 1.0  # close
        return action

    def open_gripper_action(self):
        action = self.zero_action()
        action[self.gripper_idx] = -1.0  # open
        return action

    def pull_action(self, pull_direction):
        """
        Pull the drawer open by retracting the arm (telescoping segments).

        The Stretch3 arm extends to the left. To pull a drawer open, we
        retract the telescope joints (negative arm extension action) while
        keeping the gripper closed. The base stays stationary.
        """
        action = self.zero_action()
        # Retract telescope arm segments (indices 1-4 in right_slice: arm_l3, l2, l1, l0)
        action[1] = -0.3  # arm_l3
        action[2] = -0.3  # arm_l2
        action[3] = -0.3  # arm_l1
        action[4] = -0.3  # arm_l0
        # Keep gripper closed
        action[self.gripper_idx] = 1.0
        return action


# ---------------------------------------------------------------------------
# State Machine
# ---------------------------------------------------------------------------

class ExplorerState:
    EXPLORING = "exploring"
    APPROACHING = "approaching"
    GRASPING = "grasping"
    PULLING = "pulling"
    DONE = "done"


class DrawerExplorer:
    """Main state machine: explore → detect → approach → grasp → pull."""

    def __init__(self, env, detector, grasp_client, rviz_pub=None):
        self.env = env
        self.detector = detector
        self.grasp_client = grasp_client
        self.rviz_pub = rviz_pub
        self.controller = Stretch3Controller(env)
        self.state = ExplorerState.EXPLORING

        self.target_bbox = None
        self.grasp_point = None
        self.pull_direction = None
        self.pull_start_pos = None
        self.steps_in_state = 0
        self.head_scan_phase = 0.0
        self.total_steps = 0

    def get_eef_pos(self, obs):
        """Get current end-effector position from obs."""
        key = "robot0_eef_pos"
        if key in obs:
            return obs[key]
        return np.array([0.0, 0.0, 0.5])

    def get_gripper_force(self):
        """Estimate force on gripper from sim contact data."""
        sim = self.env.sim
        # Sum contact forces on gripper bodies
        total_force = 0.0
        for i in range(sim.data.ncon):
            contact = sim.data.contact[i]
            force = np.zeros(6)
            try:
                from mujoco import mj_contactForce
                mj_contactForce(sim.model._model, sim.data._data, i, force)
                total_force += np.linalg.norm(force[:3])
            except (ImportError, AttributeError):
                # Fallback: check if force sensing available via obs
                pass
        return total_force

    def step(self, obs):
        """Execute one step of the state machine. Returns action."""
        self.steps_in_state += 1
        self.total_steps += 1

        if self.state == ExplorerState.EXPLORING:
            return self._explore_step(obs)
        elif self.state == ExplorerState.APPROACHING:
            return self._approach_step(obs)
        elif self.state == ExplorerState.GRASPING:
            return self._grasp_step(obs)
        elif self.state == ExplorerState.PULLING:
            return self._pull_step(obs)
        else:
            return self.controller.zero_action()

    def _explore_step(self, obs):
        """Random walk + head panning + check cameras for drawers."""
        # Pan head back and forth
        self.head_scan_phase += 0.02
        pan = 0.6 * np.sin(self.head_scan_phase)
        tilt = -0.2 + 0.1 * np.sin(self.head_scan_phase * 0.5)

        # Build action: random walk + head pan
        walk_action = self.controller.random_walk_action()
        head_action = self.controller.pan_head_action(pan, tilt)
        action = walk_action + head_action  # they don't overlap

        # Every 10 steps, check cameras for drawers
        if self.steps_in_state % 10 == 0:
            detection = self._check_cameras_for_drawer(obs)
            if detection is not None:
                # Check if drawer is within the robot's reachable workspace
                robot_base_pos = obs.get("robot0_base_pos", np.zeros(3))
                reachable, reason = is_drawer_reachable(detection, robot_base_pos)

                if not reachable:
                    # Publish red bbox to RViz for unreachable drawers
                    if self.rviz_pub:
                        self.rviz_pub.publish_drawer_bbox(detection, reachable=False)

                    drawer_height = detection["centroid"][2]
                    # If height is OK but just too far, navigate toward it
                    if EEF_HEIGHT_MIN <= drawer_height <= EEF_HEIGHT_MAX and "too far" in reason:
                        print(f"[Explorer] Drawer at reachable height but too far. Navigating toward it...")
                        self.target_bbox = detection
                        self.grasp_point = find_handle_grasp_point(detection)
                        self.pull_direction = detection["pull_direction"]
                        self.state = ExplorerState.APPROACHING
                        self.steps_in_state = 0
                    else:
                        print(f"[Explorer] Drawer detected but NOT reachable: {reason}. Continuing search...")
                else:
                    # Publish green bbox to RViz for reachable drawers
                    if self.rviz_pub:
                        self.rviz_pub.publish_drawer_bbox(detection, reachable=True)

                    print(f"[Explorer] Drawer detected and reachable! Transitioning to APPROACHING. (step {self.total_steps})")
                    self.target_bbox = detection
                    self.grasp_point = find_handle_grasp_point(detection)
                    self.pull_direction = detection["pull_direction"]

                    # Call ROS2 service
                    rgb_key = f"{HEAD_CAM}_image"
                    depth_key = f"{HEAD_CAM}_depth"
                    if rgb_key in obs and self.grasp_client:
                        self.grasp_client.call_service(
                            obs[rgb_key], obs.get(depth_key, np.zeros((IMG_HEIGHT, IMG_WIDTH))),
                            detection
                        )

                    self.state = ExplorerState.APPROACHING
                    self.steps_in_state = 0

        # Timeout: change direction occasionally
        if self.steps_in_state > 100:
            self.steps_in_state = 0

        return action

    def _check_cameras_for_drawer(self, obs):
        """Run drawer detection on available camera images."""
        for cam_name in [HEAD_CAM, WRIST_CAM]:
            rgb_key = f"{cam_name}_image"
            depth_key = f"{cam_name}_depth"

            if rgb_key not in obs:
                continue

            rgb = obs[rgb_key]
            if rgb.ndim == 2:
                continue

            # Ensure uint8 for detector
            if rgb.dtype != np.uint8:
                rgb = (rgb * 255).astype(np.uint8) if rgb.max() <= 1.0 else rgb.astype(np.uint8)

            detections = self.detector.detect_drawers(rgb)
            if not detections:
                continue

            best = detections[0]
            print(f"[Explorer] Detected drawer in {cam_name}: bbox={best['bbox']}, score={best['score']:.2f}")

            # Get depth and compute 3D bbox
            if depth_key in obs:
                depth = obs[depth_key]
                bbox_3d = compute_3d_bbox(self.env, depth, best["mask"], cam_name)
                if bbox_3d is not None:
                    # Publish detection image with 2D bbox to RViz
                    if self.rviz_pub:
                        self.rviz_pub.publish_detection_image(rgb, best["bbox"], reachable=True)
                        self.rviz_pub.publish_drawer_bbox(bbox_3d, reachable=True)
                    return bbox_3d

        return None

    def _approach_step(self, obs):
        """
        Position the robot so the drawer is to its LEFT at the correct lateral distance.

        The Stretch3 arm extends perpendicular to the drive direction (to the left).
        Approach strategy:
          Phase 1: Drive toward the computed parking pose (beside the drawer)
          Phase 2: Rotate so the drawer is to the robot's left at optimal distance
          Phase 3: Extend arm to reach the handle
        """
        eef_pos = self.get_eef_pos(obs)
        robot_base_pos = obs.get("robot0_base_pos", np.zeros(3))
        robot_base_quat = obs.get("robot0_base_quat", np.array([1, 0, 0, 0]))

        # Compute target parking pose (robot beside the drawer)
        target_pos_2d, target_yaw = compute_approach_pose(
            self.target_bbox["centroid"], robot_base_pos
        )

        # Current robot yaw
        w, x, y, z = robot_base_quat
        robot_yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y**2 + z**2))

        # Distance to parking position
        pos_error = target_pos_2d - robot_base_pos[:2]
        pos_dist = np.linalg.norm(pos_error)

        # Heading error to parking orientation
        yaw_error = target_yaw - robot_yaw
        yaw_error = (yaw_error + np.pi) % (2 * np.pi) - np.pi

        # EEF distance to grasp point (for final arm reach phase)
        eef_dist = np.linalg.norm(self.grasp_point - eef_pos)

        if self.steps_in_state > 300:
            print("[Explorer] Approach timeout — returning to EXPLORING.")
            self.state = ExplorerState.EXPLORING
            self.steps_in_state = 0
            return self.controller.zero_action()

        action = self.controller.zero_action()

        if pos_dist > 0.15:
            # Phase 1: Drive to parking position
            angle_to_park = np.arctan2(pos_error[1], pos_error[0])
            drive_heading_error = angle_to_park - robot_yaw
            drive_heading_error = (drive_heading_error + np.pi) % (2 * np.pi) - np.pi

            if abs(drive_heading_error) > 0.3:
                # Rotate to face parking position
                action[self.controller.base_yaw_idx] = np.clip(drive_heading_error * 0.5, -0.4, 0.4)
            else:
                # Drive forward toward parking spot
                action[self.controller.base_fwd_idx] = np.clip(pos_dist, 0.1, 0.3)
                action[self.controller.base_yaw_idx] = np.clip(drive_heading_error * 0.2, -0.15, 0.15)

        elif abs(yaw_error) > 0.15:
            # Phase 2: Rotate in place so drawer is to our left
            action[self.controller.base_yaw_idx] = np.clip(yaw_error * 0.4, -0.3, 0.3)

        elif eef_dist > 0.05:
            # Phase 3: Extend arm laterally to reach the handle
            # Verify reachability with current orientation
            reachable, reason = is_drawer_reachable(
                self.target_bbox, robot_base_pos, robot_base_quat
            )
            if not reachable:
                print(f"[Explorer] Drawer not reachable from current pose: {reason}. Adjusting...")
                # Small correction: nudge forward/backward
                action[self.controller.base_fwd_idx] = 0.1
                return action

            action = self.controller.move_to_grasp_action(eef_pos, self.grasp_point)

        else:
            # Reached the grasp point
            print(f"[Explorer] Reached grasp point. Transitioning to GRASPING. (step {self.total_steps})")
            self.state = ExplorerState.GRASPING
            self.steps_in_state = 0

        return action

    def _grasp_step(self, obs):
        """Close gripper on the drawer handle."""
        if self.steps_in_state < 5:
            # Open gripper first
            return self.controller.open_gripper_action()
        elif self.steps_in_state < 15:
            # Final approach (small forward)
            action = self.controller.zero_action()
            action[1] = 0.1  # extend arm slightly
            return action
        elif self.steps_in_state < 30:
            # Close gripper
            return self.controller.close_gripper_action()
        else:
            print(f"[Explorer] Gripper closed. Transitioning to PULLING. (step {self.total_steps})")
            self.state = ExplorerState.PULLING
            self.steps_in_state = 0
            self.pull_start_pos = self.get_eef_pos(obs).copy()
            return self.controller.zero_action()

    def _pull_step(self, obs):
        """Pull drawer outward until force threshold or max distance."""
        eef_pos = self.get_eef_pos(obs)
        pull_distance = np.linalg.norm(eef_pos - self.pull_start_pos)
        force = self.get_gripper_force()

        if force > PULL_FORCE_THRESHOLD:
            print(f"[Explorer] Force limit reached ({force:.1f} N). Drawer fully open! (step {self.total_steps})")
            self.state = ExplorerState.DONE
            return self.controller.open_gripper_action()

        if pull_distance > MAX_PULL_DISTANCE:
            print(f"[Explorer] Max pull distance reached ({pull_distance:.3f} m). (step {self.total_steps})")
            self.state = ExplorerState.DONE
            return self.controller.open_gripper_action()

        if self.steps_in_state > 300:
            print(f"[Explorer] Pull timeout. (step {self.total_steps})")
            self.state = ExplorerState.DONE
            return self.controller.open_gripper_action()

        # Pull using base backward motion while keeping gripper closed
        return self.controller.pull_action(self.pull_direction)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Stretch3 Drawer Explorer")
    print("=" * 60)

    # Create environment with camera observations + depth
    env = robosuite.make(
        "FindOpenCloseDrawer",
        robots="Stretch3",
        controller_configs=STRETCH3_CONTROLLER_CONFIG,
        has_renderer=True,
        has_offscreen_renderer=True,
        render_camera=None,
        ignore_done=True,
        use_camera_obs=True,
        camera_names=[HEAD_CAM, WRIST_CAM],
        camera_heights=[IMG_HEIGHT, IMG_HEIGHT],
        camera_widths=[IMG_WIDTH, IMG_WIDTH],
        camera_depths=[True, True],
        control_freq=20,
        renderer="mjviewer",
    )

    print(f"  Action dim: {env.action_spec[0].shape}")
    print(f"  Robot: {env.robots[0].name}")

    detector = DrawerDetector()
    grasp_client = DrawerGraspClient() if HAS_ROS2 else None
    rviz_pub = RVizPublisher() if HAS_ROS2 else None

    explorer = DrawerExplorer(env, detector, grasp_client, rviz_pub=rviz_pub)

    obs = env.reset()
    print(f"  Obs keys: {sorted(obs.keys())}")
    print(f"  Camera images available: {[k for k in obs.keys() if 'image' in k or 'depth' in k]}")
    print("\nStarting exploration loop (close viewer to exit)...")
    print("-" * 60)

    try:
        for step in range(100000):
            action = explorer.step(obs)
            obs, reward, done, info = env.step(action)
            env.render()

            # Publish camera images to ROS2 every 5 steps (10 Hz at 50 Hz sim)
            if rviz_pub and step % 5 == 0:
                rviz_pub.publish_cameras(obs)
                rviz_pub.spin_once()

            if explorer.state == ExplorerState.DONE:
                print(f"\n[DONE] Drawer task completed at step {step}.")
                print("Resetting for another attempt...")
                obs = env.reset()
                explorer = DrawerExplorer(env, detector, grasp_client, rviz_pub=rviz_pub)

            if done:
                obs = env.reset()
                explorer = DrawerExplorer(env, detector, grasp_client, rviz_pub=rviz_pub)

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        env.close()
        if grasp_client:
            grasp_client.shutdown()
        if rviz_pub:
            rviz_pub.shutdown()
        print("Done.")


if __name__ == "__main__":
    main()
