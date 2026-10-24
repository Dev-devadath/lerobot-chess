"""Classify every square as empty / white / black from the overhead camera.

First capture a reference of the EMPTY board (lighting as it will be in play):
    .venv/bin/python chess/board_state.py --ref
Then run live classification:
    .venv/bin/python chess/board_state.py
Each square shows its label and the diff score vs the reference.
Keys:  p = print the board as text   q = quit
"""

import sys
from pathlib import Path

import cv2
import numpy as np

from board import FILES, SQ, draw_patches, load_calib, open_overhead, square_patch, warp_board

REF_PATH = Path(__file__).parent / "board_ref.png"

# Occupancy = mean colour shift of the patch vs the empty reference, in Lab.
# Chroma (a,b) shift catches cream pieces on white squares (L barely moves);
# L shift catches black pieces on white squares (chroma barely moves).
# Measured live: empty squares <= 2.3; weakest chroma-only piece (pawn on a
# dark square) 7.7; black on white gives chroma ~5 but |dL| ~180 (empties <= 13).
CHROMA_THRESH = 5.0
L_THRESH = 25.0
PIXEL_DIFF = 10.0  # per-pixel Lab distance counted as "changed" (piece pixels)
WHITE_L = 120  # median L (0..255) of changed pixels above this => white piece


def capture_reference(calib, n=20):
    cap = open_overhead()
    frames = []
    for _ in range(15):  # let exposure settle
        cap.read()
    while len(frames) < n:
        ok, f = cap.read()
        if ok:
            frames.append(warp_board(f, calib["corners"]).astype(np.float32))
    cap.release()
    ref = np.mean(frames, axis=0).astype(np.uint8)
    cv2.imwrite(str(REF_PATH), ref)
    print(f"saved {REF_PATH}")


def classify(warped, ref, nadir):
    """Returns (labels, scores): 8x8 arrays indexed [file][rank] with 'E'/'W'/'B'."""
    lab = cv2.cvtColor(warped, cv2.COLOR_BGR2LAB).astype(np.float32)
    ref_lab = cv2.cvtColor(ref, cv2.COLOR_BGR2LAB).astype(np.float32)
    labels = [[None] * 8 for _ in range(8)]
    scores = np.zeros((8, 8))
    for f in range(8):
        for r in range(8):
            p = square_patch(lab, f, r, nadir)
            q = square_patch(ref_lab, f, r, nadir)
            shift = p.reshape(-1, 3).mean(0) - q.reshape(-1, 3).mean(0)
            chroma = np.linalg.norm(shift[1:])
            scores[f][r] = chroma
            if chroma < CHROMA_THRESH and abs(shift[0]) < L_THRESH:
                labels[f][r] = "E"
            else:
                diff = np.linalg.norm(p - q, axis=2)
                changed = p[..., 0][diff > PIXEL_DIFF]
                if changed.size == 0:
                    changed = p[..., 0].ravel()
                labels[f][r] = "W" if np.median(changed) > WHITE_L else "B"
    return labels, scores


def board_text(labels):
    rows = []
    for r in range(7, -1, -1):
        rows.append(f"{r + 1}  " + " ".join(labels[f][r] if labels[f][r] != "E" else "." for f in range(8)))
    rows.append("   " + " ".join(FILES))
    return "\n".join(rows)


def main():
    calib = load_calib()
    if "--ref" in sys.argv:
        capture_reference(calib)
        return
    if not REF_PATH.exists():
        sys.exit("no reference image; run with --ref on an empty board first")
    if "nadir" not in calib:
        sys.exit("no nadir in calibration; run calibrate_nadir.py first")

    ref = cv2.imread(str(REF_PATH))
    nadir = calib["nadir"]
    cap = open_overhead()

    colors = {"E": (0, 255, 255), "W": (255, 255, 255), "B": (0, 0, 0)}
    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        warped = warp_board(frame, calib["corners"])
        labels, scores = classify(warped, ref, nadir)

        disp = draw_patches(warped, nadir)
        for f in range(8):
            for r in range(8):
                x, y = f * SQ + 4, (7 - r) * SQ + SQ - 6
                txt = f"{labels[f][r]} {scores[f][r]:.0f}"
                cv2.putText(disp, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
                cv2.putText(disp, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[labels[f][r]], 1)
        cv2.imshow("board state", disp)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("p"):
            print(board_text(labels), "\n")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
