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
from openpi.policies import dual_ur_policy
from recorder import Recorder 
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
        self.left_wrist_cam_serial = cam["left_wrist_cam_serial"]
        self.right_wrist_cam_serial = cam["right_wrist_cam_serial"]
        self.exterior_cam_serial = cam["exterior_cam_serial"]
        self.cam_fps = cam.get("fps", 30)

        # Video config
        video = cfg["video"]
        self.video_fps = video.get("fps", 7)
        self.visualize = video["visualize"]

        # Robot config
        robot = cfg["robot"]
        self.left_robot_ip = robot["left_robot_ip"]
        self.right_robot_ip = robot["right_robot_ip"]
        self.left_gripper_port = robot["left_gripper_port"]
        self.right_gripper_port = robot["right_gripper_port"]
        self.gripper_reverse = robot["gripper_reverse"]
        self.left_initial_pose = robot["left_initial_pose"]
        self.right_initial_pose = robot["right_initial_pose"]
        self.initial_pose = self.left_initial_pose + self.right_initial_pose
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
        left_wrist_video = video_dir / f"left_wrist_{time_str}.mp4"
        right_wrist_video = video_dir / f"right_wrist_{time_str}.mp4"
        exterior_video = video_dir / f"exterior_{time_str}.mp4"

        # Recorder  
        self.recorder = Recorder(log_path=log_path, video_path=[left_wrist_video, right_wrist_video, exterior_video], display_fps=self.video_fps, visualize=self.visualize)
        
        # create symlink to latest log
        update_latest_symlink(log_path, latest_path)

        # create FPS counters
        self.fps_action = FpsCounter(name="action")

        # Internal states
        self.l_rtde_r = None
        self.l_rtde_c = None
        self.r_rtde_r = None
        self.r_rtde_c = None
        self.cameras = None
        self.close_threshold = 0.7
        self._velocity = 0.5 # used in moveJ, not servoj
        self._acceleration = 0.5 # used in moveJ, not servoj
        self._dt = 0.0002
        self._lookahead_time = 0.1
        self._gain = 100
        self.last_l_gpos = 1
        self.last_r_gpos = 1
        self._gripper_force = 20
        self.left_gpos = 1
        self.right_gpos = 1
        self._last_servoj_ts = None

    # --------------------------- ROBOT --------------------------- #
    def connect_robot(self):
        """Connect to UR5e robot and print current state."""
        try:
            logging.info("\n===== [ROBOT] Connecting to UR5e robot =====")
            self.l_rtde_r = RTDEReceiveInterface(self.left_robot_ip)
            self.l_rtde_c = RTDEControlInterface(self.left_robot_ip)
            self.r_rtde_r = RTDEReceiveInterface(self.right_robot_ip)
            self.r_rtde_c = RTDEControlInterface(self.right_robot_ip)

            # Joint positions
            joints = self.l_rtde_r.getActualQ() + self.r_rtde_r.getActualQ()
            if joints and len(joints) == 12:
                formatted = [round(j, 4) for j in joints]
                logging.info(f"[ROBOT] Current joint positions: {formatted}")
            else:
                logging.info("[ERROR] Failed to read joint positions.")

            # TCP pose
            tcp_pose = self.l_rtde_r.getActualTCPPose() + self.r_rtde_r.getActualTCPPose()
            if tcp_pose and len(tcp_pose) == 12:
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

            left_wrist_cfg = RealSenseCameraConfig(
                serial_number_or_name=self.left_wrist_cam_serial,
                fps=self.cam_fps,
                width=640,
                height=480,
                color_mode=ColorMode.RGB,
                use_depth=False,
                rotation=Cv2Rotation.NO_ROTATION,
            )

            right_wrist_cfg = RealSenseCameraConfig(
                serial_number_or_name=self.right_wrist_cam_serial,
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

            camera_config = {"left_wrist_image": left_wrist_cfg, "right_wrist_image": right_wrist_cfg, "exterior_image": exterior_cfg}
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
            self.left_gripper = PGE(port=self.left_gripper_port)
            self.left_gripper.init_feedback()
            self.left_gripper.set_force(self._gripper_force)
            self.right_gripper = PGE(port=self.right_gripper_port)
            self.right_gripper.init_feedback()
            self.right_gripper.set_force(self._gripper_force)
            # Start gripper state reader
            self._start_gripper_state_reader()
            logging.info("[GRIPPER] DH Gripper initialized successfully.")

        except Exception as e:
            logging.error("[ERROR] Failed to initialize DH Gripper.")
            logging.error(f"Exception: {e}\n")
            self.gripper = None

    # --------------------------- GRIPPER THREAD --------------------------- #
    def _start_gripper_state_reader(self):
        threading.Thread(target=self._read_left_gripper_state, daemon=True).start()
        threading.Thread(target=self._read_right_gripper_state, daemon=True).start()

    # --------------------------- GRIPPER WAIT --------------------------- #
    def wait_for_gripper_states(self):
        if hasattr(self.left_gripper, 'position') and hasattr(self.right_gripper, 'position'):
            while self.left_gripper.position is None or self.right_gripper.position is None:
                logging.info("[GRIPPER] Waiting for gripper state to be obtained...")
                time.sleep(0.1)
        else:
            while not hasattr(self.left_gripper, 'position') and not hasattr(self.right_gripper, 'position'):
                logging.info("[GRIPPER] Waiting for gripper position to be set...")
                time.sleep(0.1)
        
    # --------------------------- GRIPPER STATE --------------------------- #
    def _read_left_gripper_state(self):
        self.left_gripper.position = None
        while True:
            gripper_position = 0.0 if self.left_gpos <= self.close_threshold else 1.0

            if self.gripper_reverse:
                gripper_position = 1 - gripper_position

            if gripper_position != self.last_l_gpos:
                self.left_gripper.set_pos(val=int(1000 * gripper_position), blocking=False)
                self.last_l_gpos = gripper_position

            gripper_pos = self.left_gripper.read_pos() / 1000.0
            if self.gripper_reverse:
                gripper_pos = 1 - gripper_pos

            self.left_gripper.position = gripper_pos
            time.sleep(0.01)

    def _read_right_gripper_state(self):
        self.right_gripper.position = None
        while True:
            gripper_position = 0.0 if self.right_gpos <= self.close_threshold else 1.0

            if self.gripper_reverse:
                gripper_position = 1 - gripper_position

            if gripper_position != self.last_r_gpos:
                self.right_gripper.set_pos(val=int(1000 * gripper_position), blocking=False)
                self.last_r_gpos = gripper_position

            gripper_pos = self.right_gripper.read_pos() / 1000.0
            if self.gripper_reverse:
                gripper_pos = 1 - gripper_pos

            self.right_gripper.position = gripper_pos
            time.sleep(0.01)

    # --------------------------- OBS TRANSFER --------------------------- #
    def _transfer_obs_state(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """Transfer raw observation state to Dual UR policy format."""

        state = np.concatenate((
            np.asarray(obs["joint_positions"][:6], dtype=np.float32),
            np.asarray([obs["left_gripper_position"]], dtype=np.float32),
            np.asarray(obs["joint_positions"][6:], dtype=np.float32),
            np.asarray([obs["right_gripper_position"]], dtype=np.float32),
        ))

        ur_obs = {
            "observation/state": state,
            "observation/exterior_image": image_tools.convert_to_uint8(
                image_tools.resize_with_pad(obs["exterior_image"], 224, 224)
            ),
            "observation/left_wrist_image": image_tools.convert_to_uint8(
                image_tools.resize_with_pad(obs["left_wrist_image"], 224, 224)
            ),
            "observation/right_wrist_image": image_tools.convert_to_uint8(
                image_tools.resize_with_pad(obs["right_wrist_image"], 224, 224)
            ),
            "prompt": obs["prompt"],
        }

        return ur_obs

    # --------------------------- OBS STATE --------------------------- #
    def get_obs_state(self) -> Dict[str, Any]:
        """Return current observation from robot."""
        obs = {}

        # Robot state
        if self.l_rtde_r and self.r_rtde_r:
            obs["joint_positions"] = np.asarray(self.l_rtde_r.getActualQ() + self.r_rtde_r.getActualQ(), dtype=np.float32)
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
        if self.left_gripper and self.right_gripper:
            obs["left_gripper_position"] = self.left_gripper.position
            obs["right_gripper_position"] = self.right_gripper.position
            # obs["gripper_position_bin"] = 0 if self.gripper.position <= self.close_threshold else 1
        
        return self._transfer_obs_state(obs) 

    # --------------------------- ACTION EXECUTION --------------------------- #
    def execute_actions(self, actions: np.ndarray, block: bool = False):
        """Execute the inferenced actions from the model."""
        if self.l_rtde_c is None or self.r_rtde_c is None:
            logging.error("[ERROR] Robot controller not connected. Cannot execute actions.")
            return

        if block:
            logging.info("[STATE] Moving robot to initial pose...")
            self.l_rtde_c.moveJ(actions[:6], self._velocity, self._acceleration)
            self.r_rtde_c.moveJ(actions[6:], self._velocity, self._acceleration)
            logging.info("[STATE] Robot reached initial pose.")
        else:
            for i, action in enumerate(actions[:self.action_horizon]):
                start_time = time.perf_counter()

                left_qpos = action[:6].tolist()
                self.left_gpos = action[6]
                right_qpos = action[7:13].tolist()
                self.right_gpos = action[13]
                # Move robot
                t_start = self.l_rtde_c.initPeriod()
                self.l_rtde_c.servoJ(left_qpos, self._velocity, self._acceleration, self._dt, self._lookahead_time, self._gain)
                self.l_rtde_c.waitPeriod(t_start)

                t_start = self.r_rtde_c.initPeriod()
                self.r_rtde_c.servoJ(right_qpos, self._velocity, self._acceleration, self._dt, self._lookahead_time, self._gain)
                self.r_rtde_c.waitPeriod(t_start)

                elapsed = time.perf_counter() - start_time
                to_sleep = 1.0 / self.action_fps - elapsed
                if to_sleep > 0:
                    time.sleep(to_sleep)
                self.fps_action.update()

    # --------------------------- PIPELINE --------------------------- #
    def run(self):
        """Main pipeline: connect robot, cameras, and print state."""
        logging.info("========== Starting Inference Pipeline ==========")
        self.connect_robot()
        self.connect_cameras()
        self.execute_actions(self.initial_pose, block=True) # move to initial pose
        self.connect_gripper()
        self.wait_for_gripper_states()
        obs = self.get_obs_state()
        logging.info(f"[STATE] Observation state: {obs.keys()}")
        policy = _policy_config.create_trained_policy(self.model_config, self.checkpoint_dir)
        logging.info("Warming up the model")
        start = time.time()
        policy.infer(obs)
        logging.info(f"Model warmup completed, took {time.time() - start:.2f}s")
        infer_time = 1
        logging.info("========== Starting Inference Loop ==========")
        try:
            while True:
                start_time = time.perf_counter()
                obs = self.get_obs_state()
                result = policy.infer(obs)
                self.execute_actions(result["actions"])
                self.recorder.submit_actions(result["actions"][:self.action_horizon], infer_time, obs["prompt"])
                self.recorder.submit_obs(obs)
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
    config_path = Path(__file__).parent / "config" / "cfg_dual_ur.yaml"
    inference = Inference(config_path)
    inference.run()

# --------------------------- ENTRY POINT --------------------------- #
if __name__ == "__main__":
    main()
