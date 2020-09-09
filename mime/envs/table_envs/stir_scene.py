import numpy as np
import pybullet as pb  # only used for euler2quat

from .table_scene import TableScene
from .table_modder import TableModder
from ..script import Script


class StirScene(TableScene):
    def __init__(self, **kwargs):
        super(StirScene, self).__init__(**kwargs)
        self._target = None
        self._modder = TableModder(self, self._randomize)

        # linear velocity x2 for the real setup
        v, w = self._max_tool_velocity
        self._max_tool_velocity = (1.5 * v, w)

    def load(self):
        super(StirScene, self).load()

    def reset(self, np_random):
        """
        Reset the cube position and arm position.
        """
        super(StirScene, self).reset(np_random)
        modder = self._modder

        # define workspace, tool position and cube position
        low, high = self.workspace
        low, high = np.array(low.copy()), np.array(high.copy())
        low[:2] += 0.05
        high[:2] -= 0.05

        tool_pos = np_random.uniform(low=low, high=high)

        # set arm to a random position
        self.robot.arm.reset_tool(tool_pos)

        # Here's the random starting position
        self._start_stir_pos = np_random.uniform(low=low, high=high)

        # Here's the random move
        self._move_pos = np_random.uniform(low=[1.0, 1.0, 0.5], high=[2.0, 2.0, 1.0])

        # Ensure move doesn't move out of the workspace
        for i, (sp, mp) in enumerate(zip(self._start_stir_pos, self._move_pos)):
            if (ssp + mp) > high[i]:
                self._start_stir_pos[i] -= ((ssp + mp) - high[i])

        # load and set cage to a random position
        modder.load_cage(np_random)

    def script(self):
        """
        Script to generate expert demonstrations.
        """
        arm = self.robot.arm
        
        # Here so that the pattern is readable (kidna)
        ssp = self._start_stir_pos
        mp = self._move_pos
        
        # Move to:
        # start
        # start + x + z
        # start + x + y 
        # start + y + z
        # start
        stir_positions = [ ssp,
            [ssp[0] + mp[0], ssp[1], ssp[2] + mp[2]],
            [ssp[0] + mp[0], ssp[1] + mp[1], ssp[2]],
            [ssp[0], ssp[1] + mp[1], ssp[2] + mp[2]],
            ssp
        ]

        sc = Script(self)
        return [
            sc.tool_move(arm, stir_positions[0]),
            sc.tool_move(arm, stir_positions[1]),
            sc.tool_move(arm, stir_positions[2]),
            sc.tool_move(arm, stir_positions[3]),
            sc.tool_move(arm, stir_positions[4]),
        ]

    def get_reward(self, action):
        return 0

    def is_task_success(self):
        return 0


def test_scene():
    from time import sleep
    scene = StirScene(robot_type='UR5')
    scene.renders(True)
    np_random = np.random.RandomState(1)
    while True:
        scene.reset(np_random)
        sleep(1)


if __name__ == "__main__":
    test_scene()
