import time
from typing import Any, Dict, Optional

import numpy as np

from gello.cameras.camera import CameraDriver
from gello.robots.robot import Robot


class Rate:
    def __init__(self, rate: float):
        self.last = time.time()
        self.rate = rate

    def sleep(self) -> None:
        while self.last + 1.0 / self.rate > time.time():
            time.sleep(0.0001)
        self.last = time.time()


class RobotEnv:
    def __init__(
        self,
        robot: Robot,
        control_rate_hz: float = 100.0,
        camera_dict: Optional[Dict[str, CameraDriver]] = None,
    ) -> None:
        self._robot = robot
        self._rate = Rate(control_rate_hz)
        self._camera_dict = {} if camera_dict is None else camera_dict

    def robot(self) -> Robot:
        """Get the robot object.

        Returns:
            robot: the robot object.
        """
        return self._robot

    def __len__(self):
        return 0

    def step(self, joints: np.ndarray) -> Dict[str, Any]:
        """Step the environment forward.

        Args:
            joints: joint angles command to step the environment with.

        Returns:
            obs: observation from the environment.
        """
        assert self._robot.num_dofs() + 1 == len(joints), f"Expected {self._robot.num_dofs()} joint values, got {len(joints)}"
        self._robot.command_joint_state(joints)
        self._rate.sleep()
        return self.get_obs()

    def queue_policy_guidance_action(self, joints: np.ndarray) -> None:
        """Store a policy guidance action in the server buffer without applying it to the robot.

        Use this when running a teleop device in policy guidance mode. The server
        includes the action as policy_guidance_action in get_observations() results
        so a separate policy process can observe the guidance signal.
        """
        if joints is None:
            self._robot.set_policy_guidance_action(None)  # type: ignore[attr-defined]
            return
        
        expected_dofs = self._robot.num_dofs() + 1
        joints = np.asarray(joints)
        is_trajectory = joints.ndim == 3
        assert is_trajectory and joints.shape[-1] == expected_dofs, \
               f"input shape:{joints.shape}, robot dofs+1:{expected_dofs}"
        self._robot.set_policy_guidance_action(joints)  # type: ignore[attr-defined]

    def get_obs(self, render_images: bool = True) -> Dict[str, Any]:
        """Get observation from the environment.

        Returns:
            obs: observation from the environment.
        """
        observations = {}
        if render_images:
            for name, camera in self._camera_dict.items():
                image, depth = camera.read()
                observations[f"{name}_rgb"] = image
                observations[f"{name}_depth"] = depth

        if render_images:
            robot_obs = self._robot.get_observations()
        else:
            robot_obs = self._robot.get_observations(render_images=False)  # type: ignore[call-arg]
        assert "joint_positions" in robot_obs
        assert "joint_velocities" in robot_obs
        assert "ee_pos_quat" in robot_obs
        observations["joint_positions"] = robot_obs["joint_positions"]
        observations["joint_velocities"] = robot_obs["joint_velocities"]
        observations["ee_pos_quat"] = robot_obs["ee_pos_quat"]
        observations["gripper_position"] = robot_obs["gripper_position"]
        for img_key in [img_key for img_key in robot_obs.keys() if "_rgb" in img_key]:
            if img_key in robot_obs:
                observations[img_key] = robot_obs[img_key]
        if "policy_guidance_chunk" in robot_obs:
            observations["policy_guidance_chunk"] = robot_obs["policy_guidance_chunk"]
        return observations


def main() -> None:
    pass


if __name__ == "__main__":
    main()
