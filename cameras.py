"""Open cameras by fingerprint instead of by OpenCV index (macOS reshuffles
indices whenever a camera is plugged in or the Iriun phone cam reconnects).

Default stream properties, before anything is set, identify each device:
    overhead Iriun phone cam   1920 wide, 30 fps default
    MacBook built-in camera    1920 wide, 15 fps default   (never wanted)
    wrist USB2.0 PC CAMERA     640 wide

    cap = open_camera(OVERHEAD)
    cap = open_camera(WRIST)
"""

import cv2

OVERHEAD = "overhead"
WRIST = "wrist"

FINGERPRINTS = {
    OVERHEAD: dict(width=1920, min_fps=25, set_size=(1920, 1080)),
    WRIST: dict(width=640, min_fps=0, set_size=(640, 480)),
}
MAX_INDEX = 6


def probe(idx):
    cap = cv2.VideoCapture(idx)
    if not cap.isOpened():
        return None
    props = (cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    return props


def list_cameras():
    """[(index, width, default_fps)] for every openable index."""
    out = []
    for i in range(MAX_INDEX):
        p = probe(i)
        if p:
            out.append((i, int(p[0]), int(p[1])))
    return out


def camera_index(kind):
    fp = FINGERPRINTS[kind]
    cams = list_cameras()
    for i, w, fps in cams:
        if w == fp["width"] and fps >= fp["min_fps"]:
            return i
    raise RuntimeError(f"no {kind} camera found; cameras (index, width, fps): {cams}")


def open_camera(kind, fps=30, mjpg=False):
    idx = camera_index(kind)
    cap = cv2.VideoCapture(idx)
    if mjpg:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    w, h = FINGERPRINTS[kind]["set_size"]
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv2.CAP_PROP_FPS, fps)
    if not cap.isOpened():
        raise RuntimeError(f"{kind} camera (index {idx}) failed to open")
    print(f"{kind} camera: index {idx}")
    return cap


if __name__ == "__main__":
    for i, w, fps in list_cameras():
        print(f"index {i}: {w} wide, {fps} fps default")
    for kind in (OVERHEAD, WRIST):
        try:
            print(kind, "->", camera_index(kind))
        except RuntimeError as e:
            print(e)
