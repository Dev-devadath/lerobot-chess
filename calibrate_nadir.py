"""Estimate the camera nadir (point on the board directly under the camera)
from how tall pieces lean, and save it to board_calib.json.

Run calibrate_board.py first.

Setup: put two tall pieces (king, queen) on squares far apart, e.g. a8 and c5.
On the warped view click, for each piece: base centre, then top centre.
After 4 clicks the clean sample patches are drawn; move pieces around to
check each patch only ever contains its own square's piece.

Keys:  r = reset clicks   s = save   q = quit
"""

import json

import cv2

from board import CALIB_PATH, draw_grid, draw_patches, load_calib, nadir_from_leans, open_overhead, warp_board

LABELS = ["piece 1 base", "piece 1 top", "piece 2 base", "piece 2 top"]

clicks = []


def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(clicks) < 4:
        clicks.append((x, y))
        print(f"{LABELS[len(clicks) - 1]} = {clicks[-1]}")


def main():
    calib = load_calib()
    cap = open_overhead()

    cv2.namedWindow("warped")
    cv2.setMouseCallback("warped", on_mouse)
    print(f"click in order: {' -> '.join(LABELS)}   (r reset, s save, q quit)")

    nadir = None
    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        warped = warp_board(frame, calib["corners"])

        if len(clicks) == 4:
            nadir = nadir_from_leans([(clicks[0], clicks[1]), (clicks[2], clicks[3])])
            disp = draw_patches(draw_grid(warped), nadir)
        else:
            nadir = None
            disp = draw_grid(warped)
            cv2.putText(disp, f"click {LABELS[len(clicks)]}", (10, 790), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        for i, p in enumerate(clicks):
            cv2.circle(disp, p, 4, (255, 0, 255), -1)
            if i % 2 == 1:
                cv2.line(disp, clicks[i - 1], p, (255, 0, 255), 1)
        cv2.imshow("warped", disp)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("r"):
            clicks.clear()
        if key == ord("s") and nadir is not None:
            calib["nadir"] = [float(nadir[0]), float(nadir[1])]
            with open(CALIB_PATH, "w") as f:
                json.dump(calib, f, indent=2)
            cv2.imwrite(str(CALIB_PATH.parent / "board_nadir_check.png"), disp)
            print(f"saved nadir {calib['nadir']}")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
