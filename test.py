<<<<<<< HEAD
import os
import lerobot.datasets.lerobot_dataset as lerobot_dataset
import pinocchio as pin
import numpy as np

repo_id = "scylearning/pour_water_into_beaker_20251117_v01"

# ---------------------------
# Config parameters
# ---------------------------
obs_indices = "28,29,30,31,32,33,7"     # Observation state reorder indices
action_slice = "6,-1"                   # FK slice, supports multi-segment e.g., "6,-1,6,-1"
action_indices = None                   # Optional action reorder indices

# ---------------------------
# Load dataset
# ---------------------------
dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id)
dataset = lerobot_dataset.LeRobotDataset(
    repo_id,
    delta_timestamps={k: [t / dataset_meta.fps for t in range(10)] for k in "action"},
)

# ---------------------------
# State reorder
# ---------------------------
if obs_indices:
    indices = list(map(int, obs_indices.split(",")))
    print("State reorder indices:", indices)
    print("State reorder names:", [dataset_meta.names["observation.state"][i-1] for i in indices])

    # Function to reorder state according to indices
    def reorder_state(ex):
        s = ex["observation.state"]
        ex["observation.state"] = [s[i-1] for i in indices]
        return ex

    dataset.hf_dataset = dataset.hf_dataset.map(reorder_state)
    print("State reorder done.")

# ---------------------------
# Action reorder
# ---------------------------
if action_indices:
    indices = list(map(int, action_indices.split(",")))
    print("Action reorder indices:", indices)
    print("Action reorder names:", [dataset_meta.names["action"][i-1] for i in indices])

    # Function to reorder action according to indices
    def reorder_action(ex):
        a = ex["action"]
        ex["action"] = [a[i-1] for i in indices]
        return ex

    dataset.hf_dataset = dataset.hf_dataset.map(reorder_action)
    print("Action reorder done.")

# ---------------------------
# Action FK (multi-segment support)
# ---------------------------
if action_slice:
    slice_vals = list(map(int, action_slice.split(",")))
    assert len(slice_vals) % 2 == 0, "action_slice must be in pairs, e.g., 6,-1 or 6,-1,6,-1"

    print("Action FK slices:", [(slice_vals[i], slice_vals[i+1]) for i in range(0, len(slice_vals), 2)])

    # Load UR5e model
    urdf_path = "/home/suchenyu/openpi/urdf/ur5e/ur5e.urdf"
    print(f"Loading URDF model from {urdf_path}")
    model = pin.buildModelFromUrdf(urdf_path)
    data = model.createData()
    base_id = model.getFrameId("base")
    tool_id = model.getFrameId("tool0")
    print("UR5e model loaded.")

    # FK function: convert joint angles to end-effector pose
    def fk(q):
        q = np.array(q)
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        M_tool = data.oMf[tool_id]
        M_base = data.oMf[base_id]
        M = M_base.inverse() * M_tool
        p = M.translation
        rotvec = pin.log3(M.rotation)
        return [p[0], p[1], p[2], rotvec[0], rotvec[1], rotvec[2]]

    # Convert action using multi-segment FK
    def convert_action(a):
        a = list(a)
        out = []
        start_idx = 0
        for k in range(0, len(slice_vals), 2):
            i, j = slice_vals[k], slice_vals[k+1]

            print(f"Processing FK segment: start={start_idx}, i={i}, j={j}")

            # FK segment
            fk_part = fk(a[start_idx:start_idx+i])
            out.extend(fk_part)
            print(f"  FK output length: {len(fk_part)}")

            # Keep original action segment
            if j != -1:
                out.extend(a[start_idx+i:start_idx+j])
                print(f"  Keeping original actions: indices {start_idx+i} to {start_idx+j-1}")
                start_idx = start_idx + j
            else:
                # -1 means keep remaining actions
                out.extend(a[start_idx+i:])
                print(f"  Keeping remaining original actions from index {start_idx+i} onward")
                start_idx = len(a)

        return np.asarray(out, dtype=np.float32).tolist()

    def convert_action_example(ex):
        ex["action"] = convert_action(ex["action"])
        return ex

    dataset.hf_dataset = dataset.hf_dataset.map(convert_action_example)
    print("Action FK conversion done.")

print("All processing done.")
=======
#!/usr/bin/env python3
import os
import subprocess
import tempfile
from pathlib import Path
from PIL import Image
import numpy as np
import time

# ------------ 参数 ------------
width, height = 1280, 720
fps = 30
duration = 10            # 秒
num_frames = fps * duration
root = Path("ffmpeg_compare_test")
frames_dir = root / "frames"
frames_dir.mkdir(parents=True, exist_ok=True)

# ------------ 生成帧（静态背景 + 平滑移动方块） ------------
def generate_frames():
    print(f"[I] Generating {num_frames} frames ({width}x{height}) ...")
    # 生成一个略有纹理但固定的背景
    rng = np.random.RandomState(12345)
    background = (rng.randint(0, 64, (height, width, 3)).astype(np.uint8) + 200).clip(0,255).astype(np.uint8)
    for i in range(num_frames):
        frame = background.copy()
        # 平滑移动方块（x 线性，y 正弦）
        x = int((i * 6) % (width - 120))
        y = int(200 + 60 * np.sin(i / 30.0))
        frame[y:y+120, x:x+120] = [255, 30, 30]  # 红方块
        # 轻微局部噪声（模拟传感器噪声）
        if i % 10 == 0:
            frame[10:30, 10:30] = rng.randint(0, 255, (20,20,3), dtype=np.uint8)
        Image.fromarray(frame).save(frames_dir / f"frame-{i:06d}.png", optimize=True)
        if i % 300 == 0 and i > 0:
            print(f"  generated {i}/{num_frames}")
    print("[I] Frame generation done.")

# ------------ ffmpeg encode helper ------------
def ffmpeg_encode(frames_pattern, out_path, codec_args):
    # frames_pattern example: frames/frame-%06d.png
    cmd = [
        "ffmpeg",
        "-y",
        "-framerate", str(fps),
        "-i", str(frames_pattern),
        "-c:v"
    ] + codec_args + [
        "-pix_fmt", "yuv420p",
        "-g", str(fps * 2),   # keyint = 2 seconds
        "-an",
        str(out_path)
    ]
    print("[CMD]", " ".join(cmd))
    start = time.time()
    cp = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    elapsed = time.time() - start
    print(f"[I] encoding finished in {elapsed:.1f}s, returncode={cp.returncode}")
    # print encoder stderr head/tail for debugging
    out_log = cp.stderr
    head = "\n".join(out_log.splitlines()[:20])
    tail = "\n".join(out_log.splitlines()[-10:])
    print("---- ffmpeg top log ----")
    print(head)
    print("---- ffmpeg tail log ----")
    print(tail)
    return cp.returncode == 0

# ------------ 主流程 ------------
def main():
    generate_frames()
    frames_pattern = frames_dir / "frame-%06d.png"

    encoders = {
        "h264": {
            "out": root / "video_h264.mp4",
            # libx264: use veryslow to favor compression, crf 23 is default-good quality
            "args": ["libx264", "-preset", "veryslow", "-crf", "23"]
        },
        "hevc": {
            "out": root / "video_hevc.mp4",
            # libx265: slower preset, crf 28 a typical balance
            "args": ["libx265", "-preset", "slow", "-x265-params", "crf=28"]
        },
        "av1": {
            "out": root / "video_av1.mkv",
            "args": ["libaom-av1", "-cpu-used", "4", "-crf", "40", "-b:v", "0"]
        }

    }

    results = {}
    for name, cfg in encoders.items():
        print(f"\n=== Encoding {name} ===")
        ok = ffmpeg_encode(frames_pattern, cfg["out"], cfg["args"])
        if not ok:
            print(f"[ERR] encoding {name} failed. Check ffmpeg log above.")
            continue
        size = cfg["out"].stat().st_size
        results[name] = (cfg["out"], size)
        print(f"[OK] {cfg['out'].name}: {size:,} bytes ({size/1024/1024:.2f} MB)")

    print("\n=== Summary ===")
    for name, (path, size) in results.items():
        print(f" - {name:6}: {path.name} -> {size/1024/1024:.2f} MB")

    print(f"\nFrames dir: {frames_dir}")
    print("You can delete it after inspection to save disk space.")

if __name__ == "__main__":
    main()
>>>>>>> origin/main
