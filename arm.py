"""SO-101 follower helpers: connection, FK/IK (placo), slow ramped moves.

Joint angles are lerobot degrees (use_degrees=True), which the
so101_new_calib URDF is built to match: all-zero = arm stretched straight
forward along +x. Positions are metres in the arm's base_link frame.
"""

import json
import time
from pathlib import Path

import numpy as np

from lerobot.model.kinematics import RobotKinematics
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

PORT = "/dev/tty.usbmodem5B3D0453051"
ROBOT_ID = "my_follower"
URDF = Path(__file__).parent / "so101_urdf" / "so101_new_calib.urdf"
ARM_CALIB_PATH = Path(__file__).parent / "arm_calib.json"

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
# Both lerobot and the new_calib URDF put 0 deg at the middle of each joint's
# range, but the swept range never matches the URDF exactly, so a few degrees
# of offset per joint remain. urdf = lerobot - OFFSET. Fitted by fit_offsets.py.
OFFSET_PATH = Path(__file__).parent / "joint_offset.json"
OFFSET = np.array(json.load(open(OFFSET_PATH))) if OFFSET_PATH.exists() else np.zeros(5)
# URDF joint limits, degrees
AXIS_WEIGHT = 1e-4  # vs position weight 1.0 (m): 10 deg of tilt costs about as much as 1.7 mm
LIMITS = {
    "shoulder_pan": (-110, 110),
    "shoulder_lift": (-100, 100),
    "elbow_flex": (-96.8, 96.8),
    "wrist_flex": (-95, 95),
    "wrist_roll": (-157, 162),
}


def connect():
    robot = SO101Follower(SO101FollowerConfig(port=PORT, id=ROBOT_ID))
    robot.connect()
    return robot


def read_joints(robot, raw=False):
    """Arm joints in URDF degrees (no gripper), in JOINTS order. raw=True: lerobot degrees."""
    obs = robot.get_observation()
    q = np.array([obs[f"{j}.pos"] for j in JOINTS], dtype=float)
    return q if raw else q - OFFSET


def read_gripper(robot):
    return robot.get_observation()["gripper.pos"]


def send_joints(robot, q, gripper=None):
    """q in URDF degrees."""
    action = {f"{j}.pos": float(v) for j, v in zip(JOINTS, np.asarray(q, float) + OFFSET)}
    if gripper is not None:
        action["gripper.pos"] = float(gripper)
    robot.send_action(action)


TRACK_TOL_DEG = 0.3
TRACK_ITERS = 5


def move_to(robot, q_target, duration_s=2.0, hz=30, settle_s=0.4):
    """Linear ramp in joint space to q_target, then trim out steady-state error.

    Under load the servos settle a degree or two short of the command (gravity
    droop), which is ~1 cm at the tool. After the ramp, re-command the target
    plus the measured shortfall a few times. Returns the final error (deg).
    """
    q0 = read_joints(robot)
    q_target = np.asarray(q_target, float)
    n = max(1, int(duration_s * hz))
    for i in range(1, n + 1):
        send_joints(robot, q0 + (q_target - q0) * i / n)
        time.sleep(1.0 / hz)
    q_cmd = q_target.copy()
    for _ in range(TRACK_ITERS):
        time.sleep(settle_s)
        err = read_joints(robot) - q_target
        if np.abs(err).max() < TRACK_TOL_DEG:
            break
        q_cmd -= err
        send_joints(robot, q_cmd)
    time.sleep(settle_s)
    return read_joints(robot) - q_target


class Kinematics:
    """FK/IK for gripper_frame_link. In that frame +z points along the gripper
    (the direction it 'points'); at q=0 that is world +x."""

    def __init__(self):
        self.k = RobotKinematics(str(URDF), "gripper_frame_link", JOINTS)
        # A 5-DoF arm cannot hit an arbitrary full orientation, so IK constrains
        # only the pointing direction: position task + axis-align task.
        self.axis_task = self.k.solver.add_axisalign_task(
            "gripper_frame_link", np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, -1.0])
        )
        # Low weight: position wins, the gripper tilts when it must (far squares).
        self.axis_task.configure("gripper_axis", "soft", AXIS_WEIGHT)
        # Roll does not affect the pointing axis, so hold it at a fixed value
        # instead of letting it drift. Must match the roll used during touch
        # calibration, because the fitted tool offset rotates with it.
        self.roll_task = self.k.solver.add_joints_task()
        self.set_roll(0.0)
        self.roll_task.configure("wrist_roll_hold", "soft", 1e-2)

    def set_roll(self, roll_deg):
        self.roll_task.set_joints({"wrist_roll": np.deg2rad(roll_deg)})

    def fk(self, q):
        """4x4 pose of gripper_frame_link for joints q (deg)."""
        return self.k.forward_kinematics(np.asarray(q, float))

    def pointing(self, q):
        """World direction the gripper points in."""
        return self.fk(q)[:3, 2]

    def ik(self, q_init, pos, point_dir, iters=100, pos_tol=1e-3):
        """Joints (deg) putting gripper_frame_link at pos (m) pointing along point_dir.

        Iterates placo's one-step solver from q_init. Returns (q, pos_err_m,
        angle_err_deg); the pointing direction is soft, so check angle_err.
        """
        T = np.eye(4)
        T[:3, 3] = pos
        self.axis_task.targetAxis_world = np.asarray(point_dir, float) / np.linalg.norm(point_dir)
        q = np.asarray(q_init, float).copy()
        err = np.inf
        for _ in range(iters):
            q = self.k.inverse_kinematics(q, T, orientation_weight=0.0)
            err = np.linalg.norm(self.fk(q)[:3, 3] - pos)
            if err < pos_tol:
                break
        cos = np.clip(np.dot(self.pointing(q), self.axis_task.targetAxis_world), -1, 1)
        return q, err, np.degrees(np.arccos(cos))

    def tool_pos(self, q, tool):
        """World position of a point given in gripper_frame_link coordinates."""
        return (self.fk(q) @ np.array([*tool, 1.0]))[:3]

    def ik_tool(self, q_init, pos, point_dir, tool, rounds=4):
        """Like ik(), but places the tool point (gripper-frame offset) at pos.

        The frame target is pos minus the tool offset rotated into the world;
        the rotation changes a little with q, so re-solve a few rounds.
        """
        q = np.asarray(q_init, float)
        frame_target = np.asarray(pos, float)
        for _ in range(rounds):
            q, _, tilt = self.ik(q, frame_target, point_dir)
            err_vec = self.tool_pos(q, tool) - pos
            frame_target = frame_target - err_vec
        return q, np.linalg.norm(err_vec), tilt


def within_limits(q, margin_deg=0.0):
    bad = []
    for j, v in zip(JOINTS, q):
        lo, hi = LIMITS[j]
        if not (lo + margin_deg <= v <= hi - margin_deg):
            bad.append(f"{j}={v:.0f}")
    return bad


def load_arm_calib(path=ARM_CALIB_PATH):
    with open(path) as f:
        d = json.load(f)
    for k in ("a1", "file_dir", "rank_dir", "normal", "tool_offset"):
        if k in d:
            d[k] = np.array(d[k])
    return d


def square_xyz(calib, file_idx, rank_idx):
    """Centre of a square on the board surface, in arm base frame (m)."""
    return calib["a1"] + file_idx * calib["file_dir"] + rank_idx * calib["rank_dir"]
