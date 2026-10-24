"""Overlay a live camera on a training frame so it can be re-aimed to match the
recorded framing before an eval. Default is the overhead phone; --cam wrist checks
the wrist camera mount (put the arm at rest with the gripper closed: the fixed
finger must sit exactly on the green outline, since it is rigid to the camera).

    .venv/bin/python chess/align_top_cam.py                  live window, q to quit
    .venv/bin/python chess/align_top_cam.py --cam wrist
    .venv/bin/python chess/align_top_cam.py --once out.jpg   save one blended frame

The reference is the first frame of episode 49 of datasets/so101_chess_act for that
camera, extracted on first run and cached as chess/<cam>_ref_ep49.jpg.
"""
import glob
import pathlib
import sys

import cv2
import numpy as np

HERE = pathlib.Path(__file__).parent
CAM = sys.argv[sys.argv.index("--cam") + 1] if "--cam" in sys.argv else "top"
REF = HERE / f"{CAM}_ref_ep49.jpg"
DATASET = HERE.parent / "datasets/so101_chess_act"
EPISODE = 49


def reference():
    if REF.exists():
        return cv2.imread(str(REF))
    import av
    import pandas as pd
    ep = pd.concat(pd.read_parquet(f) for f in sorted(glob.glob(f"{DATASET}/meta/episodes/chunk-*/file-*.parquet")))
    r = ep[ep.episode_index == EPISODE].iloc[0]
    k = f"videos/observation.images.{CAM}"
    path = f"{DATASET}/{k}/chunk-{int(r[k + '/chunk_index']):03d}/file-{int(r[k + '/file_index']):03d}.mp4"
    target = int(round(r[k + "/from_timestamp"] * 30))
    with av.open(path) as c:
        for i, f in enumerate(c.decode(video=0)):
            if i == target:
                img = f.to_ndarray(format="bgr24")
                break
    cv2.imwrite(str(REF), img)
    return img


def main():
    sys.path.insert(0, str(HERE.parent))
    from chess.cameras import OVERHEAD, WRIST, camera_index
    ref = reference()
    cap = cv2.VideoCapture(camera_index(WRIST if CAM == "wrist" else OVERHEAD))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    once = "--once" in sys.argv
    while True:
        ok, live = cap.read()
        if not ok:
            continue
        live = cv2.resize(live, (ref.shape[1], ref.shape[0]))
        blend = cv2.addWeighted(ref, 0.5, live, 0.5, 0)
        edges = cv2.Canny(cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY), 80, 160)
        blend[edges > 0] = (0, 255, 0)          # training board outline in green
        if once:
            out = sys.argv[sys.argv.index("--once") + 1]
            cv2.imwrite(out, blend)
            print("saved", out)
            break
        cv2.imshow(f"align {CAM}: green = training frame, adjust camera until it lines up (q quits)", blend)
        if cv2.waitKey(30) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
