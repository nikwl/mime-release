from .stir_scene import StirScene
from .table_env import TableEnv
from .table_cam_env import TableCamEnv


class StirEnv(TableEnv):
    """ Pick environment, trajectory observation, linear tool control """

    def __init__(self, **kwargs):
        scene = StirScene(**kwargs)
        super(StirEnv, self).__init__(scene)

        self.observation_space = self._make_dict_space(
            'linear_velocity'
        )
        self.action_space = self._make_dict_space(
            'linear_velocity',
            'joint_velocity',
            'grip_velocity',
        )

    def _get_observation(self, scene):
        return super(StirEnv, self)._get_observation(scene)


class StirCamEnv(TableCamEnv):
    """ Pick environment, camera observation, linear tool control """

    def __init__(self, view_rand, gui_resolution, cam_resolution, num_cameras,
                 **kwargs):
        scene = StirScene(**kwargs)
        super(StirCamEnv, self).__init__(scene, view_rand, gui_resolution,
                                         cam_resolution, num_cameras)

        self.action_space = self._make_dict_space(
            'linear_velocity',
            'joint_velocity',
            'grip_velocity')

    def _get_observation(self, scene):
        return super(StirCamEnv, self)._get_observation(scene)
