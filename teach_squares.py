"""Teach the real tool position of each reachable square using the leader arm.

For each square the follower moves (under torque) to the model's guess, a few
cm above the square. Then the LEADER takes over in delta mode: the follower
moves by however much you move the leader from where it was when the square
started, so there is no jump. Nudge the closed jaw tip onto the centre of the
square, resting on the surface, then press Enter. The follower's actual tool
position is recorded as that square's point. Roll and gripper are not
followed (the tool offset is only valid at the calibration roll).

BOARD MUST BE EMPTY. Keep a hand near the power switch.

Keys (press then Enter):  <Enter> accept   s skip square   q quit and save
Usage:  .venv/bin/python chess/teach_squares.py [squares...]   default: all reachable
        .venv/bin/python chess/teach_squares.py --verify        visit taught points
"""

import json
import select
import sys
import time
from pathlib import Path

import numpy as np

from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig

from arm import JOINTS, Kinematics, connect, load_arm_calib, move_to, read_joints, send_joints, square_xyz
from board import FILES
from reach_map import HOVER_M, calib_roll, hover_point, solve_all, solve_point

LEADER_PORT = "/dev/tty.usbmodem5B610337631"
LEADER_ID = "my_leader"
POINTS_PATH = Path(__file__).parent / "square_points.json"
APPROACH_M = 0.03  # model's guess height above the board where the leader takes over
FOLLOW_HZ = 30


def read_line_nonblocking():
    if select.select([sys.stdin], [], [], 0)[0]:
        return sys.stdin.readline().strip()
    return None


def leader_joints(leader):
    a = leader.get_action()
    return np.array([a[f"{j}.pos"] for j in JOINTS], dtype=float)


def follow_until_key(robot, leader, q_start):
    """Delta teleop from q_start until a line is entered. Returns (key, q_follower)."""
    q_lead0 = leader_joints(leader)
    while True:
        delta = leader_joints(leader) - q_lead0
        delta[4] = 0.0  # keep roll at the calibration value
        send_joints(robot, q_start + delta)
        key = read_line_nonblocking()
        if key is not None:
            time.sleep(0.3)
            return key, read_joints(robot)
        time.sleep(1.0 / FOLLOW_HZ)


def main():
    kin = Kinematics()
    calib = load_arm_calib()
    tool = calib.get("tool_offset", np.zeros(3))
    roll = calib_roll(calib)
    kin.set_roll(roll)
    taught = json.load(open(POINTS_PATH)) if POINTS_PATH.exists() else {}

    if "--verify" in sys.argv:
        verify(kin, calib, tool, roll, taught)
        return

    wanted = [a for a in sys.argv[1:] if not a.startswith("--")]
    results = solve_all(kin, calib)
    reachable = [f"{FILES[f]}{r + 1}" for r in range(8) for f in range(8) if results[(f, r)][0] == "o"]
    squares = wanted or reachable
    print(f"teaching {len(squares)} squares: {' '.join(squares)}")
    if taught:
        print(f"already taught: {' '.join(s for s in taught if not s.startswith('_'))} (will be overwritten if revisited)")

    input("\nBOARD EMPTY? Enter to start... ")
    robot = connect()
    leader = SO101Leader(SO101LeaderConfig(port=LEADER_PORT, id=LEADER_ID))
    leader.connect()
    try:
        for sq in squares:
            f, r = FILES.index(sq[0]), int(sq[1]) - 1
            q_hover, err_h, _, bad_h = solve_point(kin, hover_point(calib, f, r), -calib["normal"], tool, roll)
            p_approach = square_xyz(calib, f, r) + APPROACH_M * calib["normal"]
            q_app, err_a, _, bad_a = solve_point(kin, p_approach, -calib["normal"], tool, roll, q_warm=q_hover)
            if bad_h or bad_a or err_h > 0.01 or err_a > 0.01:
                print(f"{sq}: no IK solution, skipping")
                continue
            move_to(robot, q_hover, duration_s=1.5)
            move_to(robot, q_app, duration_s=1.0)
            print(f">> {sq}: nudge the tip onto the square centre with the leader, then Enter (s skip, q quit)")
            key, q = follow_until_key(robot, leader, q_app)
            if key == "q":
                break
            if key == "s":
                continue
            p = kin.tool_pos(q, tool)
            d = (p - square_xyz(calib, f, r)) * 1000
            taught[sq] = {"xyz": p.tolist(), "joints": q.tolist()}
            print(f"   {sq}: tool xyz {np.round(p, 4)}  correction vs model {np.round(d)} mm")
            json.dump(taught, open(POINTS_PATH, "w"), indent=2)
            move_to(robot, q_hover, duration_s=1.0)  # lift before travelling
    finally:
        leader.disconnect()
        robot.disconnect()

    if taught:
        d = np.array([np.array(v["xyz"]) - square_xyz(calib, FILES.index(s[0]), int(s[1]) - 1)
                      for s, v in taught.items() if not s.startswith("_")]) * 1000
        print(f"\n{len(taught)} squares taught. correction vs model: mean {np.round(d.mean(0))} mm, "
              f"max |xy| {np.linalg.norm(d[:, :2], axis=1).max():.0f} mm, z range {d[:, 2].min():.0f}..{d[:, 2].max():.0f} mm")
        print(f"saved {POINTS_PATH}")


STEP_M = 0.002


def surface_point(taught_xyz, taught, calib):
    """A taught point lifted by the stored touch offset = the real mat surface."""
    return np.array(taught_xyz) + taught.get("_z_touch_offset", 0.0) * calib["normal"]


def verify(kin, calib, tool, roll, taught):
    """Visit each taught square 1 cm above the surface. Keys (then Enter):
    d / u  step the tip 2 mm down / up      t  tip is just touching: record it
    <Enter> next square                      q  quit
    The touch offset (mean over recorded squares) is saved into square_points.json."""
    squares = [s for s in taught if not s.startswith("_")]
    if not squares:
        sys.exit("nothing taught yet")
    print(f"current touch offset: {taught.get('_z_touch_offset', 0.0) * 1000:.0f} mm")
    input(f"BOARD EMPTY? will visit {len(squares)} taught squares. Enter to start... ")
    robot = connect()
    contacts = []
    try:
        for sq in squares:
            base = surface_point(taught[sq]["xyz"], taught, calib)
            lift = 0.01
            q_h, *_ = solve_point(kin, base + HOVER_M * calib["normal"], -calib["normal"], tool, roll)
            move_to(robot, q_h, duration_s=1.5)
            q = q_h
            while True:
                p = base + lift * calib["normal"]
                q, err, _, bad = solve_point(kin, p, -calib["normal"], tool, roll, q_warm=q)
                if bad or err > 0.005:
                    print(f"   {sq}: IK failed at lift {lift * 1000:.0f} mm")
                else:
                    move_to(robot, q, duration_s=0.6 if lift != 0.01 else 1.0)
                key = input(f"   {sq}: tip at {lift * 1000:+.0f} mm above assumed surface  [d/u/t/Enter/q] ").strip()
                if key == "d":
                    lift -= STEP_M
                elif key == "u":
                    lift += STEP_M
                elif key == "t":
                    contacts.append(lift)
                    print(f"   contact at {lift * 1000:+.0f} mm -> touch offset so far {np.mean(contacts) * 1000 + taught.get('_z_touch_offset', 0.0) * 1000:.0f} mm")
                elif key == "q":
                    raise KeyboardInterrupt
                else:
                    break
            move_to(robot, q_h, duration_s=1.0)
    except KeyboardInterrupt:
        pass
    finally:
        robot.disconnect()
    if contacts:
        taught["_z_touch_offset"] = taught.get("_z_touch_offset", 0.0) + float(np.mean(contacts))
        json.dump(taught, open(POINTS_PATH, "w"), indent=2)
        print(f"saved touch offset {taught['_z_touch_offset'] * 1000:.0f} mm from {len(contacts)} squares")


if __name__ == "__main__":
    main()
