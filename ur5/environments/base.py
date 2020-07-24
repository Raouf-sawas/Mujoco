from collections import OrderedDict
from mujoco_py import MjSim, MjRenderContextOffscreen
from mujoco_py import load_model_from_xml

from ur5.utils import SimulationError, XMLError, MujocoPyRenderer

REGISTERED_ENVS = {}


def register_env(target_class):
    REGISTERED_ENVS[target_class.__name__] = target_class


def make(env_name, *args, **kwargs):

    
    """Try to get the equivalent functionality of gym.make in a sloppy way."""
    if env_name not in REGISTERED_ENVS:
        raise Exception(
            "Environment {} not found. Make sure it is a registered environment among: {}".format(
                env_name, ", ".join(REGISTERED_ENVS)
            )
        )
    print(" REGISTERED_ENVS[env_name](*args, **kwargs)", REGISTERED_ENVS[env_name](*args, **kwargs))
    return REGISTERED_ENVS[env_name](*args, **kwargs)


class EnvMeta(type):
    """Metaclass for registering environments"""

    def __new__(meta, name, bases, class_dict):
        
        cls = super().__new__(meta, name, bases, class_dict)

        # List all environments that should not be registered here.
        _unregistered_envs = ["Ur5Env"]
        if cls.__name__ not in _unregistered_envs:
            register_env(cls)
        return cls


class MujocoEnv(metaclass=EnvMeta):
    """Initializes a Mujoco Environment."""
    
    def __init__(
        self,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_collision_mesh=False,
        render_visual_mesh=True,
        control_freq=10,
        horizon=1000,
        ignore_done=False,
        use_camera_obs=False,
        camera_name="frontview",
        camera_height=256,
        camera_width=256,
        camera_depth=False,
    ):

        print( "Initializes a Mujoco Environment.")

        self.has_renderer = has_renderer
        self.has_offscreen_renderer = has_offscreen_renderer
        self.render_collision_mesh = render_collision_mesh
        self.render_visual_mesh = render_visual_mesh
        self.control_freq = control_freq
        self.horizon = horizon
        self.ignore_done = ignore_done
        self.viewer = None
        self.model = None

        # settings for camera observations
        self.use_camera_obs = use_camera_obs
        if self.use_camera_obs and not self.has_offscreen_renderer:
            raise ValueError("Camera observations require an offscreen renderer.")
        self.camera_name = camera_name
        if self.use_camera_obs and self.camera_name is None:
            raise ValueError("Must specify camera name when using camera obs")
        self.camera_height = camera_height
        self.camera_width = camera_width
        self.camera_depth = camera_depth

        self._reset_internal()

    def initialize_time(self, control_freq):
        """
        Initializes the time constants used for simulation.
        """
        self.cur_time = 0
        self.model_timestep = self.sim.model.opt.timestep
        if self.model_timestep <= 0:
            raise XMLError("xml model defined non-positive time step")
        self.control_freq = control_freq
        if control_freq <= 0:
            raise SimulationError(
                "control frequency {} is invalid".format(control_freq)
            )
        self.control_timestep = 1. / control_freq

    def _load_model(self):
        """Loads an xml model, puts it in self.model"""
        pass

    def _get_reference(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        pass

    def reset(self):
        """Resets simulation."""
        # TODO(yukez): investigate black screen of death
        # if there is an active viewer window, destroy it
        self._destroy_viewer()
        print("0- looking for ENV")

        self._reset_internal()
        self.sim.forward()
        return self._get_observation()

    def _reset_internal(self):
        print("01- _load_model for ENV")

        """Resets simulation internal configurations."""
        # instantiate simulation from MJCF model
        self._load_model()
        print("loaded")
        self.mjpy_model = self.model.get_model(mode="mujoco_py")
        print("mjpy_model")
        self.sim = MjSim(self.mjpy_model)
        print("MjSim")
        self.initialize_time(self.control_freq)

        # create visualization screen or renderer
        if self.has_renderer and self.viewer is None:
            self.viewer = MujocoPyRenderer(self.sim)
            self.viewer.viewer.vopt.geomgroup[0] = (
                1 if self.render_collision_mesh else 0
            )
            self.viewer.viewer.vopt.geomgroup[1] = 1 if self.render_visual_mesh else 0

            # hiding the overlay speeds up rendering significantly
            self.viewer.viewer._hide_overlay = True

        elif self.has_offscreen_renderer:
            if self.sim._render_context_offscreen is None:
                render_context = MjRenderContextOffscreen(self.sim)
                self.sim.add_render_context(render_context)
            self.sim._render_context_offscreen.vopt.geomgroup[0] = (
                1 if self.render_collision_mesh else 0
            )
            self.sim._render_context_offscreen.vopt.geomgroup[1] = (
                1 if self.render_visual_mesh else 0
            )

        # additional housekeeping
        self.sim_state_initial = self.sim.get_state()
        self._get_reference()
        self.cur_time = 0
        self.timestep = 0
        self.done = False

    def _get_observation(self):
        """Returns an OrderedDict containing observations [(name_string, np.array), ...]."""
        return OrderedDict()

    def step(self, action):
        """Takes a step in simulation with control command @action."""
        if self.done:
            raise ValueError("executing action in terminated episode")

        self.timestep += 1
        self._pre_action(action)
        #print("action",action)
        end_time = self.cur_time + self.control_timestep
        #print("step:",action)
        while self.cur_time < end_time:
            self.sim.step()
            self.cur_time += self.model_timestep
        reward, done, info = self._post_action(action)
        return self._get_observation(), reward, done, info

    def _pre_action(self, action):
        """Do any preprocessing before taking an action."""
        print("_base action",action)
        self.sim.data.ctrl[:] = action

    def _post_action(self, action):
        """Do any housekeeping after taking an action."""
        reward = self.reward(action)

        # done if number of elapsed timesteps is greater than horizon
        self.done = (self.timestep >= self.horizon) and not self.ignore_done
        return reward, self.done, {}

    def reward(self, action):
        """Reward should be a function of state and action."""
        return 0

    def render(self):
        """
        Renders to an on-screen window.
        """
        self.viewer.render()

    def observation_spec(self):
        """
        Returns an observation as observation specification.

        An alternative design is to return an OrderedDict where the keys
        are the observation names and the values are the shapes of observations.
        We leave this alternative implementation commented out, as we find the
        current design is easier to use in practice.
        """
        observation = self._get_observation()
        return observation

        # observation_spec = OrderedDict()
        # for k, v in observation.items():
        #     observation_spec[k] = v.shape
        # return observation_spec

    def action_spec(self):
        """
        Action specification should be implemented in subclasses.

        Action space is represented by a tuple of (low, high), which are two numpy
        vectors that specify the min/max action limits per dimension.
        """
        raise NotImplementedError

    def reset_from_xml_string(self, xml_string):
        """Reloads the environment from an XML description of the environment."""

        # if there is an active viewer window, destroy it
        self.close()

        # load model from xml
        self.mjpy_model = load_model_from_xml(xml_string)
        print("xml_string",xml_string)

        self.sim = MjSim(self.mjpy_model)
        self.initialize_time(self.control_freq)
        if self.has_renderer and self.viewer is None:
            self.viewer = MujocoPyRenderer(self.sim)
            self.viewer.viewer.vopt.geomgroup[0] = (
                1 if self.render_collision_mesh else 0
            )
            self.viewer.viewer.vopt.geomgroup[1] = 1 if self.render_visual_mesh else 0

            # hiding the overlay speeds up rendering significantly
            self.viewer.viewer._hide_overlay = True

        elif self.has_offscreen_renderer:
            render_context = MjRenderContextOffscreen(self.sim)
            render_context.vopt.geomgroup[0] = 1 if self.render_collision_mesh else 0
            render_context.vopt.geomgroup[1] = 1 if self.render_visual_mesh else 0
            self.sim.add_render_context(render_context)

        self.sim_state_initial = self.sim.get_state()
        self._get_reference()
        self.cur_time = 0
        self.timestep = 0
        self.done = False

        # necessary to refresh MjData
        self.sim.forward()
    
    def _destroy_viewer(self):
        # if there is an active viewer window, destroy it
        if self.viewer is not None:
            self.viewer.close()  # change this to viewer.finish()?
            self.viewer = None

    def close(self):
        """Do any cleanup necessary here."""
        self._destroy_viewer()
