"""Fit per-joint angle offsets (lerobot deg -> URDF deg) from touch data.

Ground truth: the touched squares form a rigid, flat grid with a known pitch
(measure it with a ruler). The fit finds the pan/lift/elbow/wrist offsets
that make FK of the recorded joints match that grid best, then rebuilds the
board frame with the corrected FK.

Usage:  .venv/bin/python chess/fit_offsets.py <square_size_mm>
        .venv/bin/python chess/fit_offsets.py <square_size_mm> --frame-only
Needs chess/arm_calib.json from touch_calib.py with 8+ squares.
Writes chess/joint_offset.json and updates chess/arm_calib.json.
--frame-only keeps the existing joint offsets and tool offset (noisy touches
make the full fit unstable) and refits only the board frame from the touches.
"""

import json
import sys

import numpy as np
from scipy.optimize import least_squares

from arm import ARM_CALIB_PATH, JOINTS, OFFSET, OFFSET_PATH, Kinematics
from board import FILES
from touch_calib import fit_frame

MAX_OFFSET_DEG = 30.0
MAX_TOOL_M = 0.06
TILT_WEIGHT = 0.5  # 1 deg of board tilt costs as much as ~9 mm of point error


def rigid_fit_residuals(touches, square_m):
    """Residuals (N*3,) after the best rigid alignment of the ideal grid to [(square, xyz), ...]."""
    B = np.array([p for _, p in touches])
    A = np.array([[FILES.index(s[0]) * square_m, (int(s[1]) - 1) * square_m, 0.0] for s, _ in touches])
    ca, cb = A.mean(0), B.mean(0)
    H = (A - ca).T @ (B - cb)
    U, _, Vt = np.linalg.svd(H)
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    return ((A - ca) @ R.T + cb - B).ravel()


def main():
    frame_only = "--frame-only" in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if len(args) != 1:
        sys.exit(__doc__)
    square_m = float(args[0]) / 1000
    calib = json.load(open(ARM_CALIB_PATH))
    raw = [(s, np.array(q)) for s, q in calib["joints_raw"]]
    if len(raw) < 8:
        print(f"warning: only {len(raw)} touches; the fit will be poorly constrained")
    kin = Kinematics()

    # params: 4 joint offsets (deg) + tool offset (m) of the touch point in gripper_frame_link
    def fk_all(params):
        off = np.array([*params[:4], 0.0])  # roll barely moves the tip; keep its offset 0
        tool = np.array([*params[4:7], 1.0])
        return [(s, (kin.fk(q - off) @ tool)[:3].copy()) for s, q in raw]

    def plane_tilt(touches):
        """Angle (rad) between the touched plane's normal and the arm z axis."""
        P = np.array([p for _, p in touches])
        _, _, vt = np.linalg.svd(P - P.mean(0))
        return np.arccos(abs(vt[2][2]))

    def cost(params):
        touches = fk_all(params)
        # Base and board share the table: penalise any fitted tilt (TILT_WEIGHT m per rad).
        return np.append(rigid_fit_residuals(touches, square_m), TILT_WEIGHT * plane_tilt(touches))

    def report(label, params):
        r = cost(params)[:-1].reshape(-1, 3)
        d = np.linalg.norm(r, axis=1) * 1000
        print(f"{label}: rms {np.sqrt((d**2).mean()):.1f} mm, max {d.max():.1f} mm")

    if frame_only:
        if "tool_offset" not in calib:
            sys.exit("arm_calib.json has no tool_offset; run a full fit once first")
        sol_x = np.array([*OFFSET[:4], *calib["tool_offset"]])
        report("kept offsets    ", sol_x)
    else:
        x0 = np.zeros(7)
        lo = [-MAX_OFFSET_DEG] * 4 + [-MAX_TOOL_M] * 3
        hi = [MAX_OFFSET_DEG] * 4 + [MAX_TOOL_M] * 3
        report("before          ", x0)
        sol_j = least_squares(lambda p: cost([*p, 0, 0, 0]), np.zeros(4), bounds=(lo[:4], hi[:4]))
        report("joint offsets   ", [*sol_j.x, 0, 0, 0])
        sol = least_squares(cost, [*sol_j.x, 0, 0, 0], bounds=(lo, hi))
        report("+ tool offset   ", sol.x)
        sol_x = sol.x
    offset = np.array([*sol_x[:4], 0.0])
    tool = sol_x[4:7]
    print("offset (lerobot - urdf), deg: " + ", ".join(f"{j} {o:+.1f}" for j, o in zip(JOINTS, offset)))
    print(f"tool offset in gripper frame, mm: {np.round(tool * 1000, 1)}")

    pts = fk_all(sol_x)
    a1, file_dir, rank_dir, resid = fit_frame(pts)
    normal = np.cross(file_dir, rank_dir)
    normal /= np.linalg.norm(normal)
    if normal[2] < 0:
        normal = -normal
    print(f"square size: file {np.linalg.norm(file_dir) * 1000:.1f} mm, rank {np.linalg.norm(rank_dir) * 1000:.1f} mm (ruler {square_m * 1000:.1f})")
    print(f"board tilt vs arm z: {np.degrees(np.arccos(normal[2])):.1f} deg")
    print("frame residual per touch: " + ", ".join(f"{s} {r * 1000:.1f}" for s, r in resid) + " mm")

    with open(OFFSET_PATH, "w") as f:
        json.dump(offset.tolist(), f)
    calib.update(
        tool_offset=tool.tolist(),
        a1=a1.tolist(),
        file_dir=file_dir.tolist(),
        rank_dir=rank_dir.tolist(),
        normal=normal.tolist(),
        touches=[[s, p.tolist()] for s, p in pts],
    )
    with open(ARM_CALIB_PATH, "w") as f:
        json.dump(calib, f, indent=2)
    print(f"saved {OFFSET_PATH} and updated {ARM_CALIB_PATH}")


if __name__ == "__main__":
    main()
