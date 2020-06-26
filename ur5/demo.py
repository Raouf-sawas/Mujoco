
import numpy as np
import ur5 as suite
import time

if __name__ == "__main__":

    # get the list of all environments
    envs = sorted(suite.environments.ALL_ENVS)

    # print info and select an environment

    print("Welcome to Surreal Robotics Suite v{}!".format(suite.__version__))
    print(suite.__logo__)
    print("Here is a list of environments in the suite:\n")



    for k, env in enumerate(envs):
        print("[{}] {}".format(k, env))
    print()
    try:
        s = input(
            "Choose an environment to run "
            + "(enter a number from 0 to {}): ".format(len(envs) - 1)
        )
        # parse input into a number within range
        k = min(max(int(s), 0), len(envs))
    except:
        print("Input is not valid. Use 0 by default.")
        k = 0

    # initialize the task
    print("envs[k]=",envs[k])
    env = suite.make(
        envs[k],
        has_renderer=True,
        ignore_done=True,
        use_camera_obs=False,
        control_freq=100,
    )
    env.reset()
    env.viewer.set_camera(camera_id=0)

    # do visualization
    a=[0, 0, 0, 0, 0, 00,0]
    for i in range(10001):
        action = np.random.randn(env.dof)
        obs, reward, done, _ = env.step(a)
        a[2]=a[2]+0.0001
        print(a)
        env.render()
