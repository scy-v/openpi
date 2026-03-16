import os
import lerobot.datasets.lerobot_dataset as lerobot_dataset
import pinocchio as pin
import numpy as np
from pathlib import Path

repo_id = "scylearning/move_reagent_bottle_20260316_v01"

# ---------------------------
# Config parameters
# ---------------------------
obs_indices = "28,29,30,31,32,33,7"     # Observation state reorder indices
action_slice = "6,-1"                   # FK slice, supports multi-segment e.g., "6,-1,6,-1"
action_indices = "1, 2, 3, 4, 5, 6"                   # Optional action reorder indices
urdf_path = "urdf/ur5e/ur5e.urdf"   # URDF path for FK, only needed if action_slice is used
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
    current_file = Path(__file__)
    base_dir = current_file.parent.parent.parent.parent
    urdf_path = base_dir / urdf_path
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

            # FK segment
            fk_part = fk(a[start_idx:start_idx+i])
            out.extend(fk_part)

            # Keep original action segment
            if j != -1:
                out.extend(a[start_idx+i:start_idx+j])
                start_idx = start_idx + j
            else:
                # -1 means keep remaining actions
                out.extend(a[start_idx+i:])
                start_idx = len(a)

        return np.asarray(out, dtype=np.float32).tolist()

    def convert_action_example(ex):
        ex["action"] = convert_action(ex["action"])
        return ex

    dataset.hf_dataset = dataset.hf_dataset.map(convert_action_example)
    print("Action FK conversion done.")

print("All processing done.")