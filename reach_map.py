"""Which squares can the arm hover over with the gripper pointing down?

Dry run (no arm needed):   .venv/bin/python chess/reach_map.py
Visit each reachable square (CLEAR THE BOARD FIRST):
                           .venv/bin/python chess/reach_map.py --move          (Enter between squares)
                           .venv/bin/python chess/reach_map.py --move --auto   (no pauses; saves an
                           overhead snapshot per square to chess/hover_snaps/)

Map legend:  o reachable   x IK position error > tolerance   L joint limit hit
             t reachable but gripper tilted more than TILT_TOL from vertical
"""

import sys
import time
from pathlib import Path

import cv2
import numpy as np

from arm import JOINTS, Kinematics, connect, load_arm_calib, move_to, read_joints, square_xyz, within_limits
from board import FILES, draw_grid, load_calib, open_overhead, warp_board

SNAP_DIR = Path(__file__).parent / "hover_snaps"

HOVER_M = 0.08  # height above the board surface
POS_TOL_M = 0.005
TILT_TOL_DEG = 30.0
# IK seeds (lift, elbow, wrist): from folded-near to stretched-far. The local
# solver gets stuck in joint limits from a single seed, so try all and keep the best.
SEEDS = [(-60, 60, 60), (-20, 30, 80), (0, 0, 90), (30, -30, 90), (60, -70, 95), (80, -90, 80)]


def hover_point(calib, file_idx, rank_idx):
    return square_xyz(calib, file_idx, rank_idx) + HOVER_M * calib["normal"]


def calib_roll(calib):
    """Wrist roll (URDF deg) used during touch calibration; the tool offset is valid for it."""
    from arm import OFFSET

    rolls = [q[4] for _, q in calib.get("joints_raw", [])]
    return float(np.mean(rolls)) - OFFSET[4] if rolls else 0.0


def solve_point(kin, pos, point_dir, tool, roll, q_warm=None):
    """Best IK over all seeds, pan pre-aimed at the target. Returns (q, err, tilt, bad).

    q_warm: an extra full seed, e.g. the solution for a nearby point."""
    pan = -np.degrees(np.arctan2(pos[1], pos[0]))  # +pan swings towards -y
    seeds = [np.array([pan, lift, elbow, wrist, roll]) for lift, elbow, wrist in SEEDS]
    if q_warm is not None:
        seeds.insert(0, np.asarray(q_warm, float))
    best = None
    for seed in seeds:
        q, err, tilt = kin.ik_tool(seed, pos, point_dir, tool)
        bad = within_limits(q)
        cost = err + (0.1 if bad else 0) + 1e-4 * tilt
        if best is None or cost < best[0]:
            best = (cost, q, err, tilt, bad)
    return best[1:]


def solve_all(kin, calib):
    tool = calib.get("tool_offset", np.zeros(3))
    roll = calib_roll(calib)
    kin.set_roll(roll)
    print(f"tool offset {np.round(tool * 1000)} mm, wrist roll held at {roll:.0f} deg")
    results = {}
    for f in range(8):
        for r in range(8):
            q, err, tilt, bad = solve_point(kin, hover_point(calib, f, r), -calib["normal"], tool, roll)
            if bad:
                status = "L"
            elif err > POS_TOL_M:
                status = "x"
            elif tilt > TILT_TOL_DEG:
                status = "t"
            else:
                status = "o"
            results[(f, r)] = (status, q, err, tilt, bad)
    return results


def print_map(results):
    for r in range(7, -1, -1):
        print(f"{r + 1}  " + " ".join(results[(f, r)][0] for f in range(8)))
    print("   " + " ".join(FILES))
    n_ok = sum(1 for v in results.values() if v[0] == "o")
    print(f"\n{n_ok}/64 reachable at {HOVER_M * 100:.0f} cm hover")
    for (f, r), (status, q, err, tilt, bad) in sorted(results.items()):
        if status != "o":
            print(f"  {FILES[f]}{r + 1}: err {err * 1000:.0f} mm  tilt {tilt:.0f} deg  {' '.join(bad)}")


def main():
    kin = Kinematics()
    calib = load_arm_calib()
    results = solve_all(kin, calib)
    print_map(results)

    if "--move" not in sys.argv:
        return

    auto = "--auto" in sys.argv
    cap = None
    if auto:
        cam = load_calib()
        cap = open_overhead()
        SNAP_DIR.mkdir(exist_ok=True)
    input("\nboard clear? arm will visit every reachable square. Enter to start, Ctrl-C to stop... ")
    robot = connect()
    try:
        for r in range(8):
            for f in range(8):
                status, q, err, tilt, _ = results[(f, r)]
                if status != "o":
                    continue
                name = f"{FILES[f]}{r + 1}"
                print(f"{name}: {np.round(q, 1)}")
                track_err = move_to(robot, q, duration_s=2.0)
                actual = kin.tool_pos(read_joints(robot), calib.get("tool_offset", np.zeros(3)))
                target = hover_point(calib, f, r)
                print(
                    f"   tracking error per joint {np.round(track_err, 1)} deg"
                    f" -> tool {np.linalg.norm(actual - target) * 1000:.0f} mm from target"
                )
                if auto:
                    time.sleep(1.0)  # settle
                    for _ in range(5):  # flush buffered frames
                        cap.read()
                    ok, frame = cap.read()
                    if ok:
                        cv2.imwrite(str(SNAP_DIR / f"{name}.png"), draw_grid(warp_board(frame, cam["corners"])))
                else:
                    input("   Enter for next square... ")
    finally:
        robot.disconnect()
        if cap is not None:
            cap.release()


if __name__ == "__main__":
    main()
