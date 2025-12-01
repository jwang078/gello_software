import pickle
import threading
from typing import Any, Dict

import numpy as np
import zmq

from gello.robots.robot import Robot

DEFAULT_ROBOT_PORT = 6000


class ZMQServerRobot:
    def __init__(
        self,
        robot: Robot,
        port: int = DEFAULT_ROBOT_PORT,
        host: str = "127.0.0.1",
    ):
        self._robot = robot
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REP)
        addr = f"tcp://{host}:{port}"
        debug_message = f"Robot Sever Binding to {addr}, Robot: {robot}"
        print(debug_message)
        self._timout_message = f"Timeout in Robot Server, Robot: {robot}"
        self._socket.bind(addr)
        self._stop_event = threading.Event()

    def serve(self) -> None:
        """Serve the leader robot state over ZMQ."""
        self._socket.setsockopt(zmq.RCVTIMEO, 1000)  # Set timeout to 1000 ms
        while not self._stop_event.is_set():
            try:
                # Wait for next request from client
                message = self._socket.recv()
                request = pickle.loads(message)

                # Call the appropriate method based on the request
                method = request.get("method")
                args = request.get("args", {})
                result: Any
                if method == "num_dofs":
                    result = self._robot.num_dofs()
                elif method == "get_joint_state":
                    result = self._robot.get_joint_state()
                elif method == "command_joint_state":
                    result = self._robot.command_joint_state(**args)
                elif method == "get_observations":
                    result = self._robot.get_observations()
                else:
                    result = {"error": "Invalid method"}
                    print(result)
                    raise NotImplementedError(
                        f"Invalid method: {method}, {args, result}"
                    )

                self._socket.send(pickle.dumps(result))
            except zmq.Again:
                # Timeout occurred - don't spam the console
                pass

    def stop(self) -> None:
        """Signal the server to stop serving."""
        self._stop_event.set()


class ZMQClientRobot(Robot):
    """A class representing a ZMQ client for a leader robot."""

    def __init__(self, port: int = DEFAULT_ROBOT_PORT, host: str = "127.0.0.1"):
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REQ)
        self._socket.connect(f"tcp://{host}:{port}")

    def num_dofs(self) -> int:
        """Get the number of joints in the robot.

        Returns:
            int: The number of joints in the robot.
        """
        request = {"method": "num_dofs"}
        send_message = pickle.dumps(request)
        self._socket.send(send_message)
        result = pickle.loads(self._socket.recv())
        return result

    def get_joint_state(self) -> np.ndarray:
        """Get the current state of the leader robot.

        Returns:
            T: The current state of the leader robot.
        """
        request = {"method": "get_joint_state"}
        send_message = pickle.dumps(request)
        try:
            self._socket.send(send_message)
            result = pickle.loads(self._socket.recv())
            if isinstance(result, dict) and "error" in result:
                raise RuntimeError(result["error"])
            return result
        except zmq.Again:
            raise RuntimeError("ZMQ timeout - robot may be disconnected")
        
    def get_ee_pos(self) -> np.ndarray:
        """Get the current end effector position of the leader robot.

        Returns:
            T: The current ee state of the leader robot.
        """
        request = {"method": "get_ee_pos"}
        send_message = pickle.dumps(request)
        try:
            self._socket.send(send_message)
            result = pickle.loads(self._socket.recv())
            if isinstance(result, dict) and "error" in result:
                raise RuntimeError(result["error"])
            return result
        except zmq.Again:
            raise RuntimeError("ZMQ timeout - robot may be disconnected")

    def command_joint_state(self, joint_state: np.ndarray) -> None:
        """Command the leader robot to the given state.

        Args:
            joint_state (T): The state to command the leader robot to.
        """
        request = {
            "method": "command_joint_state",
            "args": {"joint_state": joint_state},
        }
        send_message = pickle.dumps(request)
        self._socket.send(send_message)
        result = pickle.loads(self._socket.recv())
        return result
    
    def teleport_joint_state(self, joint_state: np.ndarray) -> None:
        """Command the leader robot to the given state.

        Args:
            joint_state (T): The state to command the leader robot to.
        """
        request = {
            "method": "teleport_joint_state",
            "args": {"joint_state": joint_state},
        }
        send_message = pickle.dumps(request)
        self._socket.send(send_message)
        result = pickle.loads(self._socket.recv())
        return result

    def set_object_pose(self, object_name: str, position: np.ndarray, orientation: np.ndarray, use_gravity: bool = True) -> None:
        """Set the pose of an object in the robot's environment.

        Args:
            object_name (str): The name of the object.
            position (np.ndarray): The position of the object.
            orientation (np.ndarray): The orientation of the object.
        """
        request = {
            "method": "set_object_pose",
            "args": {
                "object_name": object_name,
                "position": position.tolist(),
                "orientation": orientation.tolist(),
                "use_gravity": use_gravity,
            },
        }
        send_message = pickle.dumps(request)
        self._socket.send(send_message)
        result = pickle.loads(self._socket.recv())
        return result
    
    def create_object(self, object_name: str, object_config: dict[str, Any], use_gravity: bool = True) -> None:
        """
        Create a new object in the robot's environment.

        Args:
            object_name (str): The name of the object.
            object_config (dict[str, Any]): Config for the object describing appearance, position, orientation, etc.
            use_gravity (bool): Whether or not to apply gravity to the object.
        """
        assert type(object_config) == dict
        assert type(object_name) == str
        assert type(use_gravity) == bool
        object_config["use_fixed_base"] = not use_gravity
        request = {
            "method": "create_object",
            "args": {
                "object_name": object_name,
                "object_config": object_config,
            }
        }
        send_message = pickle.dumps(request)
        self._socket.send(send_message)
        result = pickle.loads(self._socket.recv())
        return result
    
    def delete_object(self, object_name: str) -> None:
        """
        Deletes object from the robot environment.

        Args:
            object_name (str): The name of the object.
        """
        assert type(object_name) == str
        request = {
            "method": "delete_object",
            "args": {
                "object_name": object_name
            }
        }
        send_message = pickle.dumps(request)
        self._socket.send(send_message)
        result = pickle.loads(self._socket.recv())
        return result
    
    def clear_temp_objects(self) -> None:
        """
        Deletes objects from the robot environment that weren't there on initialization.
        """
        request = {
            "method": "clear_temp_objects",
            "args": {},
        }
        send_message = pickle.dumps(request)
        self._socket.send(send_message)
        result = pickle.loads(self._socket.recv())
        return result

    def get_observations(self) -> Dict[str, np.ndarray]:
        """Get the current observations of the leader robot.

        Returns:
            Dict[str, np.ndarray]: The current observations of the leader robot.
        """
        request = {"method": "get_observations"}
        send_message = pickle.dumps(request)
        try:
            self._socket.send(send_message)
            result = pickle.loads(self._socket.recv())
            if isinstance(result, dict) and "error" in result:
                raise RuntimeError(result["error"])
            return result
        except zmq.Again:
            raise RuntimeError("ZMQ timeout - robot may be disconnected")
        
    def disable_rendering(self) -> Dict[str, np.ndarray]:
        """Disable rendering from splat

        Returns:
            Dict[str, np.ndarray]: The current observations of the leader robot.
        """
        request = {"method": "disable_rendering"}
        send_message = pickle.dumps(request)
        try:
            self._socket.send(send_message)
            result = pickle.loads(self._socket.recv())
            if isinstance(result, dict) and "error" in result:
                raise RuntimeError(result["error"])
            return result
        except zmq.Again:
            raise RuntimeError("ZMQ timeout - robot may be disconnected")

    def enable_rendering(self) -> Dict[str, np.ndarray]:
        """Enable rendering from splat

        Returns:
            Dict[str, np.ndarray]: The current observations of the leader robot.
        """
        request = {"method": "enable_rendering"}
        send_message = pickle.dumps(request)
        try:
            self._socket.send(send_message)
            result = pickle.loads(self._socket.recv())
            if isinstance(result, dict) and "error" in result:
                raise RuntimeError(result["error"])
            return result
        except zmq.Again:
            raise RuntimeError("ZMQ timeout - robot may be disconnected")

    def close(self) -> None:
        """Close the ZMQ socket and context."""
        self._socket.close()
        self._context.term()
