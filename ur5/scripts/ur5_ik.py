"""End-effector control for bimanual Baxter robot.

This script shows how to use inverse kinematics solver from Bullet
to command the end-effectors of two arms of the Baxter robot.
"""

import os
import numpy as np

import ur5
from ur5.wrappers import IKWrapper


if __name__ == "__main__":

    # initialize a Baxter environment
    env = ur5.make(
        "Ur5Lift",
        ignore_done=True,
        has_renderer=True,
        gripper_visualization=True,
        use_camera_obs=False,
    )
    env = IKWrapper(env)

    obs = env.reset()

    # rotate the gripper so we can see it easily
    env.set_robot_joint_positions([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    bullet_data_path = os.path.join(ur5.models.assets_root, "bullet_data")

    def robot_jpos_getter():
        return np.array(env._joint_positions)

    x=0
    for t in range(100000):
        x=x+0.00001
        dpos = np.array([0,0,x])
        dquat = np.array([0.5, 0, 0, 1])
        grasp = 1.
        action = np.concatenate([dpos, dquat, [grasp]])

        obs, reward, done, info = env.step(action)
        env.render()

        if done:
            break
