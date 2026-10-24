"""Torque off; continuously print joint angles and the FK gripper position.

Use it to measure arm.OFFSET: hold the whole arm straight out horizontally
(upper arm, forearm and gripper in one straight line, gripper pointing
forward, jaws vertical). The RAW lerobot readings in that pose are OFFSET.
Once OFFSET is set, the URDF readings in that pose should be about 0 and the
gripper at about x=0.39, z=0.23.

Usage:  .venv/bin/python chess/read_pose.py      (Ctrl-C to stop)
"""

import time

import numpy as np

from arm import Kinematics, connect, read_joints

np.set_printoptions(precision=1, suppress=True)


def main():
    kin = Kinematics()
    robot = connect()
    input("HOLD THE ARM - torque is about to be released. Enter when ready... ")
    robot.bus.disable_torque()
    try:
        while True:
            raw = read_joints(robot, raw=True)
            q = read_joints(robot)
            p = kin.fk(q)[:3, 3]
            print(f"\rRAW lerobot {raw}   URDF {q}   gripper xyz {np.round(p, 3)} m   ", end="", flush=True)
            time.sleep(0.2)
    except KeyboardInterrupt:
        print()
    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()
