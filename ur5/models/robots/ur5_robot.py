import numpy as np
from ur5.models.robots.robot import Robot
from ur5.utils.mjcf_utils import xml_path_completion, array_to_string


class Ur5(Robot):
    """Ur5 is a sensitive single-arm robot designed by Franka."""

    def __init__(self):
        super().__init__(xml_path_completion("robots/ur5/robot.xml"))

        self.bottom_offset = np.array([0, 0, -0.913])
       

    def set_base_xpos(self, pos):
        """Places the robot on position @pos."""
        node = self.worldbody.find("./body[@name='base']")
        node.set("pos", array_to_string(pos - self.bottom_offset))

    @property
    def dof(self):
        return 6

    @property
    def joints(self):
        return ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]

    @property
    def init_qpos(self):
        return np.array([2, -1, 0, 1, 1,2])