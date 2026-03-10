import logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
import yaml
import os
import time
import threading
import numpy as np
from pathlib import Path
from pyDHgripper import PGE
from typing import Dict, Any
from utils import FpsCounter
from openpi_client import image_tools
from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface
from openpi.policies import ur5e_policy
from recorder_ur import Recorder 
from openpi.training import config as _config
from openpi.policies import policy_config as _policy_config
from lerobot.cameras.configs import ColorMode, Cv2Rotation
from lerobot.cameras.realsense.camera_realsense import RealSenseCameraConfig
from lerobot.cameras import make_cameras_from_configs
home = Path.home()

def update_latest_symlink(target: Path, link_name: Path):
    if link_name.exists() or link_name.is_symlink():
        link_name.unlink()
    os.symlink(target, link_name)

class Inference:
    def __init__(self, config_path: Path):
        # Load YAML config
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)

        # Model config
        model = cfg["model"]
        self.model_config = _config.get_config(model["name"])
        self.checkpoint_dir = home / model["checkpoint_dir"]
        
        # Camera config
        cam = cfg["cameras"]
        self.wrist_cam_serial = cam["wrist_cam_serial"]
        self.exterior_cam_serial = cam["exterior_cam_serial"]
        self.cam_fps = cam.get("fps", 30)

        # Video config
        video = cfg["video"]
        self.video_fps = video.get("fps", 7)
        self.visualize = video["visualize"]

        # Robot config
        robot = cfg["robot"]
        self.robot_ip = robot["ip"]
        self.gripper_port = robot["gripper_port"]
        self.gripper_reverse = robot["gripper_reverse"]
        self.initial_pose = robot["initial_pose"]
        self.action_fps = robot["action_fps"]
        self.action_horizon = robot["action_horizon"]
        # Task config
        task = cfg["task"]
        self.task_description = task["description"]
        
        # time stamps
        time_str = time.strftime('%Y%m%d-%H%M%S')
        time_path = time.strftime('%Y%m%d')

        # base dir
        base_dir = Path(__file__).parent
        log_dir = base_dir / "logs"
        video_dir = base_dir / "videos" / time_path

        # create dir
        (log_dir / "all_logs").mkdir(parents=True, exist_ok=True)
        video_dir.mkdir(parents=True, exist_ok=True)

        # log paths
        latest_path = log_dir / "latest.yaml"
        log_path = log_dir / "all_logs" / f"log_{time_str}.yaml"

        # video paths
        wrist_video = video_dir / f"wrist_{time_str}.mp4"
        exterior_video = video_dir / f"exterior_{time_str}.mp4"

        # Recorder  
        self.recorder = Recorder(log_path=log_path, video_path=[wrist_video, exterior_video], display_fps=self.video_fps, visualize=self.visualize)
        
        # create symlink to latest log
        update_latest_symlink(log_path, latest_path)

        # create FPS counters
        self.fps_action = FpsCounter(name="action")

        # Internal states
        self.rtde_r = None
        self.rtde_c = None
        self.cameras = None
        self.close_threshold = 0.7
        self._velocity = 0.5 # used in moveJ, not servoj
        self._acceleration = 0.5 # used in moveJ, not servoj
        self._dt = 0.0002
        self._lookahead_time = 0.1
        self._gain = 100
        self._last_gripper_position = 1
        self._gripper_force = 20
        self._gripper_position = 1
        self._last_servoj_ts = None

    # --------------------------- ROBOT --------------------------- #
    def connect_robot(self):
        """Connect to UR5e robot and print current state."""
        try:
            logging.info("\n===== [ROBOT] Connecting to UR5e robot =====")
            self.rtde_r = RTDEReceiveInterface(self.robot_ip)
            self.rtde_c = RTDEControlInterface(self.robot_ip)

            # Joint positions
            joints = self.rtde_r.getActualQ()
            if joints and len(joints) == 6:
                formatted = [round(j, 4) for j in joints]
                logging.info(f"[ROBOT] Current joint positions: {formatted}")
            else:
                logging.info("[ERROR] Failed to read joint positions.")

            # TCP pose
            tcp_pose = self.rtde_r.getActualTCPPose()
            if tcp_pose and len(tcp_pose) == 6:
                formatted_pose = [round(p, 4) for p in tcp_pose]
                logging.info(f"[ROBOT] Current TCP pose: {formatted_pose}")
                logging.info(
                    f"[ROBOT] Translation (m): x={formatted_pose[0]}, y={formatted_pose[1]}, z={formatted_pose[2]}"
                )
                logging.info(
                    f"[ROBOT] Rotation (rad): rx={formatted_pose[3]}, ry={formatted_pose[4]}, rz={formatted_pose[5]}"
                )
                logging.info("===== [ROBOT] UR5e initialized successfully =====\n")
            else:
                logging.info("[ERROR] Failed to read TCP pose.")

        except Exception as e:
            logging.error("===== [ERROR] Failed to connect to UR5e robot =====")
            logging.error(f"Exception: {e}\n")
            exit(1)
    # --------------------------- CAMERAS --------------------------- #
    def connect_cameras(self):
        """Initialize and connect RealSense cameras."""
        try:
            logging.info("\n===== [CAM] Initializing cameras =====")

            wrist_cfg = RealSenseCameraConfig(
                serial_number_or_name=self.wrist_cam_serial,
                fps=self.cam_fps,
                width=640,
                height=480,
                color_mode=ColorMode.RGB,
                use_depth=False,
                rotation=Cv2Rotation.NO_ROTATION,
            )

            exterior_cfg = RealSenseCameraConfig(
                serial_number_or_name=self.exterior_cam_serial,
                fps=self.cam_fps,
                width=640,
                height=480,
                color_mode=ColorMode.RGB,
                use_depth=False,
                rotation=Cv2Rotation.NO_ROTATION,
            )

            camera_config = {"wrist_image": wrist_cfg, "exterior_image": exterior_cfg}
            self.cameras = make_cameras_from_configs(camera_config)

            for name, cam in self.cameras.items():
                cam.connect()
                logging.info(f"[CAM] {name} connected successfully.")

            logging.info("===== [CAM] Cameras initialized successfully =====\n")

        except Exception as e:
            logging.error("[ERROR] Failed to initialize cameras.")
            logging.error(f"Exception: {e}\n")
            self.cameras = None

    # --------------------------- GRIPPER --------------------------- #
    def connect_gripper(self):
        """Initialize and connect the DH gripper."""
        try:
            logging.info("\n===== [GRIPPER] Initializing DH Gripper =====")
            self.gripper = PGE(port=self.gripper_port)
            self.gripper.init_feedback()
            self.gripper.set_force(self._gripper_force)
            # Start gripper state reader
            self._start_gripper_state_reader()
            logging.info("[GRIPPER] DH Gripper initialized successfully.")

        except Exception as e:
            logging.error("[ERROR] Failed to initialize DH Gripper.")
            logging.error(f"Exception: {e}\n")
            self.gripper = None

    # --------------------------- GRIPPER THREAD --------------------------- #
    def _start_gripper_state_reader(self):
        threading.Thread(target=self._read_gripper_state, daemon=True).start()

    # --------------------------- GRIPPER WAIT --------------------------- #
    def wait_for_gripper_states(self):
        if hasattr(self.gripper, 'position'):
            while self.gripper.position is None:
                logging.info("[GRIPPER] Waiting for gripper state to be obtained...")
                time.sleep(0.1)
        else:
            while not hasattr(self.gripper, 'position'):
                logging.info("[GRIPPER] Waiting for gripper position to be set...")
                time.sleep(0.1)
        
    # --------------------------- GRIPPER STATE --------------------------- #
    def _read_gripper_state(self):
        self.gripper.position = None
        while True:
            gripper_position = 0.0 if self._gripper_position <= self.close_threshold else 1.0

            if self.gripper_reverse:
                gripper_position = 1 - gripper_position

            if gripper_position != self._last_gripper_position:
                self.gripper.set_pos(val=int(1000 * gripper_position), blocking=False)
                self._last_gripper_position = gripper_position

            gripper_pos = self.gripper.read_pos() / 1000.0
            if self.gripper_reverse:
                gripper_pos = 1 - gripper_pos

            self.gripper.position = gripper_pos
            time.sleep(0.01)

    # --------------------------- OBS TRANSFER --------------------------- #
    def _transfer_obs_state(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """Transfer raw observation state to UR5e policy format."""

        state = np.concatenate((
            np.asarray(obs["joint_positions"], dtype=np.float32),
            # np.asarray(obs["tcp_pose"][:5], dtype=np.float32),
            np.asarray([obs["gripper_position"]], dtype=np.float32),
        ))

        ur5e_obs = {
            "observation/state": state,
            "observation/image": image_tools.convert_to_uint8(
                image_tools.resize_with_pad(obs["exterior_image"], 224, 224)
            ),
            "observation/wrist_image": image_tools.convert_to_uint8(
                image_tools.resize_with_pad(obs["wrist_image"], 224, 224)
            ),
            "prompt": obs["prompt"],
        }

        return ur5e_obs

    # --------------------------- OBS STATE --------------------------- #
    def get_obs_state(self) -> Dict[str, Any]:
        """Return current observation from robot."""
        obs = {}

        # Robot state
        if self.rtde_r:
            obs["joint_positions"] = np.asarray(self.rtde_r.getActualQ())
            # obs["tcp_pose"] = np.asarray(self.rtde_r.getActualTCPPose())

        # Camera images
        if self.cameras:
            for name, cam in self.cameras.items():
                frame = cam.read()
                obs[name] = frame

        # Task description    
        if self.task_description:
            obs["prompt"] = self.task_description

        # Gripper state
        if self.gripper:
            obs["gripper_position"] = self.gripper.position
            # obs["gripper_position_bin"] = 0 if self.gripper.position <= self.close_threshold else 1
        
        return self._transfer_obs_state(obs) 

    # --------------------------- ACTION EXECUTION --------------------------- #
    def execute_actions(self, actions: np.ndarray, block: bool = False):
        """Execute the inferenced actions from the model."""
        if self.rtde_c is None:
            logging.error("[ERROR] Robot controller not connected. Cannot execute actions.")
            return

        if block:
            logging.info("[STATE] Moving robot to initial pose...")
            self.rtde_c.moveJ(actions, self._velocity, self._acceleration)
            logging.info("[STATE] Robot reached initial pose.")
        else:
            for i, action in enumerate(actions[:self.action_horizon]):
                start_time = time.perf_counter()

                joint_positions = action[:6].tolist()
                # Move robot
                t_start = self.rtde_c.initPeriod()
                self.rtde_c.servoJ(joint_positions, self._velocity, self._acceleration, self._dt, self._lookahead_time, self._gain)
                self.rtde_c.waitPeriod(t_start)
                self._gripper_position = action[6]

                elapsed = time.perf_counter() - start_time
                to_sleep = 1.0 / self.action_fps - elapsed
                if to_sleep > 0:
                    time.sleep(to_sleep)
                self.fps_action.update()

    # --------------------------- PIPELINE --------------------------- #
    def run(self):
        """Main pipeline: connect robot, cameras, and print state."""
        policy = _policy_config.create_trained_policy(self.model_config, self.checkpoint_dir)
        logging.info("Warming up the model")
        start = time.time()
        obs = ur5e_policy.make_ur5e_example()
        policy.infer(obs)
        logging.info(f"Model warmup completed, took {time.time() - start:.2f}s")
        infer_time = 1
        logging.info("========== Starting Inference Loop ==========")
        try:
            while True:
                start_time = time.perf_counter()
                time.sleep(0.1)  # Small sleep to prevent CPU overload
                end_time = time.perf_counter()
                logging.info(f"[STATE] Inference loop rate: {1 / (end_time - start_time):.1f} HZ")
                infer_time += 1
        except KeyboardInterrupt:
            logging.info("[INFO] KeyboardInterrupt detec ted. Saving recorded videos before exiting...")

        except Exception as e:
            logging.error(f"[ERROR] Inference loop encountered an error: {e}")

        try:
            ans = input("Save recorded videos before exiting? [Y/n]: ").strip().lower()
            if ans in ("", "y", "yes"):
                logging.info("[INFO] Saving recorded videos before exiting...")
                self.recorder.save_video()
        except Exception as e:
            logging.error(f"[ERROR] Failed to save videos: {e}")

# --------------------------- MAIN --------------------------- #
def main():
    config_path = Path(__file__).parent / "config" / "cfg_ur.yaml"
    inference = Inference(config_path)
    inference.run()

# --------------------------- ENTRY POINT --------------------------- #
if __name__ == "__main__":
    main()
