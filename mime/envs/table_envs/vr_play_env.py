from .vr_play_scene import VRPickScene
from .table_env import TableEnv
from .table_cam_env import TableCamEnv


class VRPlayEnv(TableEnv):
    """ Pick environment, trajectory observation, linear tool control """

    def __init__(self, **kwargs):
        scene = VRPickScene(**kwargs)
        super(VRPlayEnv, self).__init__(scene)

        self.observation_space = self._make_dict_space(
            'linear_velocity',
            'grip_forces', 
            'grip_width')
        self.action_space = self._make_dict_space(
            'linear_velocity',
            'angular_velocity',
            'grip_velocity',
        )

    def _get_observation(self, scene):
        return super(VRPlayEnv, self)._get_observation(scene)

class VRPlayCamEnv(TableCamEnv):
    """ Pick environment, camera observation, linear tool control """

    def __init__(self, view_rand, gui_resolution, cam_resolution, num_cameras,
                 **kwargs):
        scene = VRPickScene(**kwargs)
        super(VRPlayCamEnv, self).__init__(scene, view_rand, gui_resolution,
                                         cam_resolution, num_cameras)

        self.action_space = self._make_dict_space(
            'linear_velocity',
            'angular_velocity',
            'grip_velocity',
        )

    def _get_observation(self, scene):
        return super(VRPlayCamEnv, self)._get_observation(scene)    