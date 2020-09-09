import numpy as np
import pybullet as pb  # only used for euler2quat

from .table_scene import TableScene
from .table_modder import TableModder
from ..script import Script


class VRPickScene(TableScene):
    def __init__(self, **kwargs):
        super(VRPickScene, self).__init__(**kwargs)
        self._modder = TableModder(self, self._randomize)

        # linear velocity x2 for the real setup
        v, w = self._max_tool_velocity
        self._max_tool_velocity = (1.5 * v, w)

        self._object_workspace = [[0.3, -0.3, 0.08], [0.6, 0.3, 0.25]]
        low, high = self._object_workspace
        self._cube_size_range = {'low': 0.05, 'high': 0.08}

        self.meshes = {
            'bowl': {
                # 'type': 'shapenet_02880940',
                'type': 'bowl/model_normalized.urdf',
                'size_range': [0.3, 0.33],
                'color': (1, 1, 1, 1),
                'mass': 1.0,
                'position_range': [low, high]  
            },
            'cup': {
                # 'type': 'shapenet_02880940',
                'type': 'dinnerware/cup/cup.urdf',
                'size_range': [0.9, 0.99],
                'color': (1, 1, 1, 1),
                'mass': 1.0,
                'position_range': [low, high]  
            },
            'cube1': {
                'type': 'cube',
                'size_range': [0.05, 0.06],
                'color': (0, 1, 0, 1),
                'mass': 0.2,
                'position_range': [low, high] 
            },
            'cube2': {
                'type': 'cube',
                'size_range': [0.05, 0.06],
                'color': (0, 1, 1, 1),
                'mass': 0.2,
                'position_range': [low, high] 
            },
            'cube3': {
                'type': 'cube',
                'size_range': [0.05, 0.06],
                'color': (1, 1, 0, 1),
                'mass': 0.2,
                'position_range': [low, high] 
            },
        }

        # We want the workspace to be as big as possible
        # x - depth
        # y - width
        # z - height
        self._workspace =  [[-0.55, -0.55, -0.1], [0.8, 0.55, 1.0]]

    def load(self):
        super(VRPickScene, self).load()

    def reset(self, np_random):
        """
        Reset the cube position and arm position.
        """
        super(VRPickScene, self).reset(np_random)
        modder = self._modder

        # Reset success state
        self.set_task_success(None)

        # load and set cage to a random position
        modder.load_cage(np_random)

        # Object workspace
        low, high = self._object_workspace

        # Place the arm somewhere in the object workspace
        tool_pos = np_random.uniform(low=low, high=high)
        self.robot.arm.reset_tool(tool_pos)

        # remove meshes present in the scene
        for mesh in self.meshes.values():
            if 'body' in mesh:
                mesh['body'].remove()

        # Spawn meshes
        meshes = self.meshes
        for name, mesh in meshes.items():
            size_range = {
                'low': mesh['size_range'][0],
                'high': mesh['size_range'][1]
            }
            mesh['body'], mesh['size'] = modder.load_mesh(
                    mesh['type'], size_range, np_random, mass=mesh['mass'])
            mesh['body'].color = mesh['color']
            pos = np_random.uniform(low=mesh['position_range'][0], high=mesh['position_range'][1])
            mesh['body'].position = (pos[0], pos[1], mesh['size'] / 2)

    def get_reward(self, action):
        return 0

    def set_task_success(self, task_successful):
        self._task_successful = task_successful

    def is_task_success(self):   
        return self._task_successful


def test_scene():
    from time import sleep
    scene = VRPickScene(robot_type='UR5')
    scene.renders(True)
    np_random = np.random.RandomState(1)
    while True:
        scene.reset(np_random)
        sleep(1)


if __name__ == "__main__":
    test_scene()
