"""Move one piece between two taught squares with the gripper.

Sequence: hover over FROM (gripper open) -> descend to grasp height -> close
-> lift -> hover over TO -> descend -> open -> lift.

Usage:  .venv/bin/python chess/pick_place.py d2 d4 [--grasp-h 20] [--open 60] [--close 0]
        .venv/bin/python chess/pick_place.py d2 d4 --dry        IK only, no motion
        .venv/bin/python chess/pick_place.py --gripper-test     open/close in place
Heights in mm above the mat surface (taught point + touch offset).
Gripper is lerobot's 0-100 scale: 0 closed, 100 fully open.
"""

import argparse
import json
import time

import numpy as np

from arm import Kinematics, connect, load_arm_calib, move_to, read_joints, send_joints
from reach_map import HOVER_M, calib_roll, solve_point
from teach_squares import POINTS_PATH, surface_point


def plan(kin, calib, tool, roll, taught, sq, grasp_h):
    """(q_hover, q_grasp) for a taught square, or raises."""
    if sq not in taught:
        raise SystemExit(f"{sq} is not taught (run teach_squares.py {sq})")
    surface = surface_point(taught[sq]["xyz"], taught, calib)
    n = calib["normal"]
    q_h, err_h, _, bad_h = solve_point(kin, surface + HOVER_M * n, -n, tool, roll)
    q_g, err_g, _, bad_g = solve_point(kin, surface + grasp_h * n, -n, tool, roll, q_warm=q_h)
    for name, err, bad in (("hover", err_h, bad_h), ("grasp", err_g, bad_g)):
        if bad or err > 0.005:
            raise SystemExit(f"{sq} {name}: IK failed (err {err * 1000:.0f} mm {bad})")
    return q_h, q_g


def set_gripper(robot, value, wait_s=0.7):
    send_joints(robot, read_joints(robot), gripper=value)
    time.sleep(wait_s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("squares", nargs="*", help="FROM TO")
    ap.add_argument("--grasp-h", type=float, default=20.0, help="mm above the surface to close at")
    ap.add_argument("--open", type=float, default=60.0)
    ap.add_argument("--close", type=float, default=0.0)
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--gripper-test", action="store_true")
    a = ap.parse_args()

    if a.gripper_test:
        robot = connect()
        try:
            print("gripper now:", robot.get_observation()["gripper.pos"])
            for v in (a.open, a.close, a.open):
                set_gripper(robot, v, wait_s=1.5)
                print(f"commanded {v:.0f} -> reads {robot.get_observation()['gripper.pos']:.0f}")
        finally:
            robot.disconnect()
        return

    if len(a.squares) != 2:
        ap.error("need FROM and TO squares")
    src, dst = a.squares
    kin = Kinematics()
    calib = load_arm_calib()
    tool = calib.get("tool_offset", np.zeros(3))
    roll = calib_roll(calib)
    kin.set_roll(roll)
    taught = json.load(open(POINTS_PATH))

    grasp_h = a.grasp_h / 1000
    q_h1, q_g1 = plan(kin, calib, tool, roll, taught, src, grasp_h)
    q_h2, q_g2 = plan(kin, calib, tool, roll, taught, dst, grasp_h + 0.003)  # release slightly higher
    print(f"{src} hover {np.round(q_h1, 1)}\n{src} grasp {np.round(q_g1, 1)}")
    print(f"{dst} hover {np.round(q_h2, 1)}\n{dst} place {np.round(q_g2, 1)}")
    if a.dry:
        return

    input(f"\nmove {src} -> {dst}, grasp at {a.grasp_h:.0f} mm. Only that piece on the board? Enter to go... ")
    robot = connect()
    try:
        set_gripper(robot, a.open)
        move_to(robot, q_h1, duration_s=2.0)
        move_to(robot, q_g1, duration_s=1.2)
        set_gripper(robot, a.close)
        print("gripper reads", round(robot.get_observation()["gripper.pos"]), "(stalls on the piece above --close)")
        move_to(robot, q_h1, duration_s=1.2)
        move_to(robot, q_h2, duration_s=2.0)
        move_to(robot, q_g2, duration_s=1.2)
        set_gripper(robot, a.open)
        move_to(robot, q_h2, duration_s=1.2)
        print("done")
    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()
