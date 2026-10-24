"""Click the four corners of the board on the live overhead feed and save them.

Usage:  .venv/bin/python chess/calibrate_board.py

Click the OUTER corners of the 64-square playing area (not the lettered
border) in this order:  a1 -> h1 -> h8 -> a8.
After 4 clicks a second window shows the warped board with a labelled grid;
move a piece to check the squares line up.

Keys:  r = reset clicks   s = save   q = quit
"""

from pathlib import Path

import cv2

from board import draw_grid, open_overhead, save_calib, warp_board

CHECK_PNG = Path(__file__).parent / "board_calib_check.png"
DISPLAY_SCALE = 0.5  # 1080p frame is too big for a laptop screen
LABELS = ["a1", "h1", "h8", "a8"]

clicks = []


def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(clicks) < 4:
        clicks.append((x / DISPLAY_SCALE, y / DISPLAY_SCALE))
        print(f"{LABELS[len(clicks) - 1]} = ({clicks[-1][0]:.0f}, {clicks[-1][1]:.0f})")


def main():
    cap = open_overhead()

    cv2.namedWindow("camera")
    cv2.setMouseCallback("camera", on_mouse)
    print(f"click corners in order: {' -> '.join(LABELS)}   (r reset, s save, q quit)")

    while True:
        ok, frame = cap.read()
        if not ok:
            continue

        disp = cv2.resize(frame, None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE)
        for i, (x, y) in enumerate(clicks):
            p = (int(x * DISPLAY_SCALE), int(y * DISPLAY_SCALE))
            cv2.circle(disp, p, 5, (0, 0, 255), -1)
            cv2.putText(disp, LABELS[i], (p[0] + 6, p[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        if len(clicks) == 4:
            pts = [(int(x * DISPLAY_SCALE), int(y * DISPLAY_SCALE)) for x, y in clicks]
            for i in range(4):
                cv2.line(disp, pts[i], pts[(i + 1) % 4], (0, 255, 0), 1)
        else:
            cv2.putText(disp, f"click {LABELS[len(clicks)]}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.imshow("camera", disp)

        if len(clicks) == 4:
            cv2.imshow("warped", draw_grid(warp_board(frame, clicks)))

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("r"):
            clicks.clear()
            cv2.destroyWindow("warped")
        if key == ord("s") and len(clicks) == 4:
            data = save_calib(clicks)
            cv2.imwrite(str(CHECK_PNG), draw_grid(warp_board(frame, clicks)))
            print(f"saved {data}")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
