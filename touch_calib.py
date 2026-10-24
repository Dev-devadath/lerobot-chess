"""Measure the board's position in the arm frame by touching squares.

Torque is disabled; you move the arm by hand. For each prompted square rest
the CLOSED gripper tip on the centre of the square, gripper pointing straight
down, and HOLD STILL. The reading is captured automatically once the arm has
not moved for STILL_S seconds (no keyboard, so nothing sags while you reach
for it). Then lift the arm clearly away; the next square is armed once the
arm has moved MOVE_DEG.

Saves chess/arm_calib.json: a1 origin, per-square file/rank vectors, board
normal, square size, and the raw joint readings (lerobot degrees). The frame
is a least-squares fit over all touches; squares may repeat (repeatability
check). Use squares the arm reaches comfortably.

Usage:  .venv/bin/python chess/touch_calib.py [squares...]   default: a1 h1 a4 h4
"""

import json
import sys
import time

import numpy as np

from arm import ARM_CALIB_PATH, OFFSET, Kinematics, connect, read_joints
from board import FILES

DEFAULT_SQUARES = ["a1", "h1", "a4", "h4"]
STILL_S = 1.0  # seconds without motion before capture
STILL_DEG = 0.6  # max joint change counted as "still"
MOVE_DEG = 15.0  # joint change needed after a capture before the next one arms


def fit_frame(touches):
    """Least-squares a1 + f*file_dir + r*rank_dir over [(square, xyz), ...]."""
    A = np.array([[1, FILES.index(s[0]), int(s[1]) - 1] for s, _ in touches])
    X = np.array([p for _, p in touches])
    coef, *_ = np.linalg.lstsq(A, X, rcond=None)
    a1, file_dir, rank_dir = coef
    resid = [(s, np.linalg.norm(A[i] @ coef - X[i])) for i, (s, _) in enumerate(touches)]
    return a1, file_dir, rank_dir, resid


def wait_still(robot):
    """Block until the joints have not changed for STILL_S; return the reading."""
    hist = []
    while True:
        q = read_joints(robot, raw=True)
        now = time.time()
        hist.append((now, q))
        hist = [(t, h) for t, h in hist if now - t <= STILL_S]
        if now - hist[0][0] >= STILL_S * 0.95 and max(np.abs(h - q).max() for _, h in hist) < STILL_DEG:
            return q
        time.sleep(0.05)


def wait_moved(robot, q_ref):
    while np.abs(read_joints(robot, raw=True) - q_ref).max() < MOVE_DEG:
        time.sleep(0.05)


def main():
    squares = sys.argv[1:] or DEFAULT_SQUARES
    if len(set(squares)) < 3:
        sys.exit("need at least 3 distinct squares")
    kin = Kinematics()
    robot = connect()
    input("HOLD THE ARM - torque is about to be released and it will go limp. Enter when ready... ")
    robot.bus.disable_torque()
    print("torque OFF - move the arm by hand. Captures happen automatically when you hold still.\n")

    touches, joints = [], []
    try:
        q_last = read_joints(robot, raw=True)
        for i, sq in enumerate(squares):
            # also before the first square: the arm starts still in its rest pose
            print("   lift away..." if i > 0 else "   move the arm to start...", end="", flush=True)
            wait_moved(robot, q_last)
            print(f"\r>> {sq}: rest the closed tip on the centre and hold still          ", end="", flush=True)
            q_raw = wait_still(robot)
            q_last = q_raw
            p = kin.fk(q_raw - OFFSET)[:3, 3].copy()
            touches.append((sq, p))
            joints.append((sq, q_raw))
            print(f"\a\r  {sq}: joints {np.round(q_raw, 1)}  ->  xyz {np.round(p, 4)} m")
    finally:
        robot.disconnect()

    a1, file_dir, rank_dir, resid = fit_frame(touches)
    normal = np.cross(file_dir, rank_dir)
    normal /= np.linalg.norm(normal)
    if normal[2] < 0:
        normal = -normal

    print(f"\nsquare size: file {np.linalg.norm(file_dir) * 1000:.1f} mm, rank {np.linalg.norm(rank_dir) * 1000:.1f} mm")
    print(f"file/rank angle: {np.degrees(np.arccos(np.dot(file_dir, rank_dir) / np.linalg.norm(file_dir) / np.linalg.norm(rank_dir))):.1f} deg (want 90)")
    print(f"board tilt vs arm z: {np.degrees(np.arccos(normal[2])):.1f} deg")
    print("fit residual per touch: " + ", ".join(f"{s} {r * 1000:.1f} mm" for s, r in resid))
    for sq in dict.fromkeys(squares):
        reps = [p for s, p in touches if s == sq]
        if len(reps) > 1:
            spread = np.linalg.norm(np.array(reps) - np.mean(reps, axis=0), axis=1) * 1000
            print(f"repeatability {sq} ({len(reps)} touches): mean dev {spread.mean():.1f} mm, max {spread.max():.1f} mm")

    data = {
        "a1": a1.tolist(),
        "file_dir": file_dir.tolist(),
        "rank_dir": rank_dir.tolist(),
        "normal": normal.tolist(),
        "touches": [[s, p.tolist()] for s, p in touches],
        "joints_raw": [[s, q.tolist()] for s, q in joints],  # lerobot degrees, before OFFSET
    }
    if ARM_CALIB_PATH.exists():  # fitted by fit_offsets.py; keep it, pick_place needs it
        old = json.load(open(ARM_CALIB_PATH))
        if "tool_offset" in old:
            data["tool_offset"] = old["tool_offset"]
    with open(ARM_CALIB_PATH, "w") as f:
        json.dump(data, f, indent=2)
    print(f"saved {ARM_CALIB_PATH}")


if __name__ == "__main__":
    main()
