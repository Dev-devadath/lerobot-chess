# SO-101 Chess Robot — Project Context

## Goal
Make an SO-101 arm play physical chess against a human: see the board, pick a move, and move the pieces properly.

## Hardware
- SO-101 follower arm (plays) + leader arm (for recording demos)
- 6x Feetech STS3215 servos; 5 joints + gripper
- Cameras:
  - **Overhead** USB webcam on a rigid stand: reads the board (required)
  - **Wrist** 32x32 mm UVC module: guides grasping (needed for the learned policy)
- PC with GPU for training (or a cloud GPU)
- 3D-printed board and pieces: large squares (5–6 cm), weighted pieces with a grippable neck

## Software stack
- **LeRobot** (Hugging Face): arm control, teleop, recording, training
- **placo**: IK/FK from the SO-101 URDF
- **OpenCV**: camera capture, calibration, board warp
- **python-chess**: board state + move legality
- **Stockfish**: picks the robot's move

## System loop
```
Overhead cam -> detect changed squares -> infer human move (python-chess)
-> Stockfish reply -> arm executes move -> repeat
```

## Board reading
- Start from the known opening position.
- Classify each square as empty / white / black / occluded. No piece-type recognition.
- The move = which square emptied + which filled. python-chess tracks piece types.
- Warp the overhead image to a top-down view with a 4-point or ArUco calibration.

## Arm control: hybrid
| Phase | Done by |
|---|---|
| Hover above source square | IK |
| Descend + grab + lift | Learned policy (or IK at first) |
| Travel to target square | IK |
| Descend + release + lift | Learned policy (or IK at first) |
| Return home | IK |

- **Square to XYZ:** `x = a1_x + file*sq`, `y = a1_y + rank*sq`. Calibrate a1 and the square size once.
- **Per-piece table:** grab height and gripper opening for P, N, B, R, Q, K.
- Always lift to a safe height (~15 cm) before moving sideways.
- **Capture:** remove the target piece to a bin first, then make the move.
- **Castling:** two moves (king, then rook).
- Optional per-square XY offsets to correct servo sag at the far squares.

## Learned pick/place policy
- Skills: `pick` and `place`; task text e.g. "pick up the knight".
- **Recording demos:** IK brings the arm to hover, then the human does only the descend/grab/lift with the leader arm.
- Vary the demos: off-centre pieces, crowded neighbours, all piece types, all board regions.
- **Model order to try:** ACT -> SmolVLA -> pi0.5 (cloud GPU).
- **Data:** start with 50–100 demos per skill, then coach (robot runs, human takes over on failure).
- **Skill done when:** gripper closed/open, back at hover height, and it descended at least 3 cm.
- Crop camera images to the board area; undistort wide lenses (ChArUco calibration).

## Build order
1. Assemble and calibrate the arm (`lerobot-calibrate`), check the cameras (`lerobot-find-cameras opencv`).
2. Calibrate the board; test IK hover over all 64 squares.
3. Play full games with pure-IK grabs; measure the failure rate.
4. Add board detection + Stockfish loop.
5. If grabs fail too often: add the wrist cam, record pick/place demos, train, and swap in the policy.

## References
- Kripner, π0.5 + IK hybrid, public datasets (HF user `Kripi`): https://kripner.com/posts/chess-robot/
- Robby Fischer, 3 cams + OpenCV + Stockfish: https://github.com/CKacha/Robby-Fischer
- lerobot-llm-chess, pure IK + LLM tool calls: https://github.com/BranavanSelvasingham/lerobot-llm-chess
- SO-ARM100 repo, camera mounts: https://github.com/TheRobotStudio/SO-ARM100