"""Board calibration: warp the overhead camera image to a top-down 8x8 view.

Calibration file (board_calib.json) holds the pixel coords of the four outer
corners of the 64-square playing area in the order a1, h1, h8, a8.
The warped image is a standard diagram view: a1 bottom-left, h8 top-right.
"""

import json
from pathlib import Path

import cv2
import numpy as np

CALIB_PATH = Path(__file__).parent / "board_calib.json"
WARP_SIZE = 800  # px, square
SQ = WARP_SIZE // 8

FILES = "abcdefgh"


def load_calib(path=CALIB_PATH):
    with open(path) as f:
        return json.load(f)


def open_overhead():
    """The overhead camera, opened by name (indices reshuffle on macOS)."""
    from cameras import OVERHEAD, open_camera

    return open_camera(OVERHEAD)


def save_calib(corners, path=CALIB_PATH):
    data = {
        "corners": [[float(x), float(y)] for x, y in corners],
        "corner_order": ["a1", "h1", "h8", "a8"],
        "warp_size": WARP_SIZE,
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    return data


def warp_matrix(corners):
    src = np.float32(corners)
    # a1 bottom-left, h1 bottom-right, h8 top-right, a8 top-left
    dst = np.float32([[0, WARP_SIZE], [WARP_SIZE, WARP_SIZE], [WARP_SIZE, 0], [0, 0]])
    return cv2.getPerspectiveTransform(src, dst)


def warp_board(frame, corners):
    return cv2.warpPerspective(frame, warp_matrix(corners), (WARP_SIZE, WARP_SIZE))


def square_roi(warped, file_idx, rank_idx):
    """Pixel crop of a square. file_idx 0..7 = a..h, rank_idx 0..7 = 1..8."""
    x0 = file_idx * SQ
    y0 = (7 - rank_idx) * SQ
    return warped[y0 : y0 + SQ, x0 : x0 + SQ]


# Pieces lean away from the camera nadir (parallax), so a neighbour's top can
# spill into a square. The base never leaves its square, so sample a patch
# shifted away from the nadir: it holds the base and dodges the spill.
PATCH_SHIFT = 0.22  # fraction of a square
PATCH_HALF = 0.22


def square_patch_rect(file_idx, rank_idx, nadir):
    """(x0, y0, x1, y1) of the clean sample patch for a square, in warped px."""
    cx = file_idx * SQ + SQ / 2
    cy = (7 - rank_idx) * SQ + SQ / 2
    d = np.array([cx, cy]) - np.array(nadir, dtype=float)
    d /= np.linalg.norm(d) + 1e-9
    px, py = cx + PATCH_SHIFT * SQ * d[0], cy + PATCH_SHIFT * SQ * d[1]
    h = PATCH_HALF * SQ
    return int(px - h), int(py - h), int(px + h), int(py + h)


def square_patch(warped, file_idx, rank_idx, nadir):
    x0, y0, x1, y1 = square_patch_rect(file_idx, rank_idx, nadir)
    return warped[y0:y1, x0:x1]


def nadir_from_leans(leans):
    """Least-squares intersection of base->top lines. leans = [((bx,by),(tx,ty)), ...]."""
    A = np.zeros((2, 2))
    b = np.zeros(2)
    for base, top in leans:
        base, top = np.array(base, float), np.array(top, float)
        d = base - top
        d /= np.linalg.norm(d)
        P = np.eye(2) - np.outer(d, d)  # projector onto the line's normal
        A += P
        b += P @ base
    return np.linalg.solve(A, b)


def draw_patches(warped, nadir):
    out = warped.copy()
    for f in range(8):
        for r in range(8):
            x0, y0, x1, y1 = square_patch_rect(f, r, nadir)
            cv2.rectangle(out, (x0, y0), (x1, y1), (0, 255, 255), 1)
    cv2.circle(out, tuple(int(v) for v in nadir), 8, (255, 0, 0), -1)
    return out


def draw_grid(warped):
    out = warped.copy()
    for i in range(1, 8):
        cv2.line(out, (i * SQ, 0), (i * SQ, WARP_SIZE), (0, 0, 255), 1)
        cv2.line(out, (0, i * SQ), (WARP_SIZE, i * SQ), (0, 0, 255), 1)
    for f in range(8):
        for r in range(8):
            cv2.putText(
                out,
                f"{FILES[f]}{r + 1}",
                (f * SQ + 4, (7 - r) * SQ + 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 0, 255),
                1,
            )
    return out
