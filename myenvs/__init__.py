# TODO: add the following commented environments into the register
# from BaxterReacherv0 import *
# from myenvs.robosuite.robosuite import *

import copy
from .registration import register, make, registry, spec


register(
    id='FetchThrowDice-v0',
    entry_point='myenvs.fetch:FetchThrowDiceEnv',
    kwargs={},
    max_episode_steps=50,
)
register(
    id='UR5PickAndPlace-v1',
    entry_point='myenvs.robosuite.UR5e:UR5eLift',
    kwargs={},
    max_episode_steps=50,
)


register(
    id='Catcher3d-v0',
    entry_point='myenvs.fetch.moving_target.catch:Catcher3dEnv',
    max_episode_steps=50,
    kwargs={},
 )

register(
    id='DyReach-v0',
    entry_point='myenvs.fetch.moving_target.dyreach:DyReachEnv',
    max_episode_steps=50,
    kwargs={'direction': (1, 0, 0),
            'velocity': 0.011}
)

register(
    id='DyCircle-v0',
    entry_point='myenvs.fetch.moving_target.dycircle:DyCircleEnv',
    max_episode_steps=50,
    kwargs={'velocity': 0.05,
            'center_offset': [1.3, 0.7, 0]}
)
register(
    id='DyPush-v0',
    entry_point='myenvs.fetch.moving_target.dypush:DyPushEnv',
    max_episode_steps=50,
    kwargs={
        'velocity': 0.011
})
for reward_type in ['sparse', 'dense']:
    suffix = 'Dense' if reward_type == 'dense' else ''
    kwargs = {
        'reward_type':reward_type,
    }

    for i in range(2, 101):
        _kwargs = copy.deepcopy(kwargs)
        _kwargs["n_bits"] = i
        register(
            id='FlipBit{}{:d}-v0'.format(suffix, i),
            entry_point='myenvs.toy:FlipBit',
            kwargs=_kwargs,
            max_episode_steps = i,
        )

    for i in range(2, 51):
        _kwargs = copy.deepcopy(kwargs)
        _kwargs["layout"] = (i, i)
        _kwargs["max_steps"] = 2 * i - 2
        register(
            id='EmptyMaze{}{:d}-v0'.format(suffix, i),
            entry_point='myenvs.toy:EmptyMaze',
            kwargs=_kwargs,
            max_episode_steps=_kwargs["max_steps"],
        )

    register(
        id='FourRoom{}-v0'.format(suffix),
        entry_point='myenvs.toy:FourRoomMaze',
        kwargs=kwargs,
        max_episode_steps=32,
    )

    register(
        id='FetchReachDiscrete{}-v0'.format(suffix),
        entry_point='myenvs.fetch:FetchReachDiscrete',
        kwargs=kwargs,
        max_episode_steps=50,
    )

    register(
        id='FetchPushDiscrete{}-v0'.format(suffix),
        entry_point='myenvs.fetch:FetchPushDiscrete',
        kwargs=kwargs,
        max_episode_steps=50,
    )

    register(
        id='FetchSlideDiscrete{}-v0'.format(suffix),
        entry_point='myenvs.fetch:FetchSlideDiscrete',
        kwargs=kwargs,
        max_episode_steps=50,
    )

    register(
        id='FetchThrow{}-v0'.format(suffix),
        entry_point='myenvs.fetch:FetchThrowEnv',
        kwargs=kwargs,
        max_episode_steps=50,
    )

    register(
        id='FetchThrowRubberBall{}-v0'.format(suffix),
        entry_point='myenvs.fetch:FetchThrowRubberBallEnv',
        kwargs=kwargs,
        max_episode_steps=50,
    )

    register(
        id='FetchPickAndThrow{}-v0'.format(suffix),
        entry_point='myenvs.fetch:FetchPickAndThrowEnv',
        kwargs=kwargs,
        max_episode_steps=50,
    )

    register(
        id='BaxterPickAndPlace{}-v0'.format(suffix),
        entry_point='myenvs.baxter.pick_and_place:BaxterPickAndPlaceEnv',
        kwargs=kwargs,
        max_episode_steps=50,
    )
    register(
        id='BaxterSlide{}-v0'.format(suffix),
        entry_point='myenvs.baxter.slide:BaxterSlideEnv',
        kwargs=kwargs,
        max_episode_steps=50,
    )
    register(
        id='BaxterPush{}-v0'.format(suffix),
        entry_point='myenvs.baxter.push:BaxterPushEnv',
        kwargs=kwargs,
        max_episode_steps=50,
    )
    register(
        id='BaxterKitting{}-v0'.format(suffix),
        entry_point='myenvs.baxter.kitting:BaxterKittingEnv',
        kwargs=kwargs,
        max_episode_steps=50,
    )
    register(
        id='BaxterReach{}-v0'.format(suffix),
        entry_point='myenvs.baxter.reach:BaxterReachEnv',
        kwargs=kwargs,
        max_episode_steps=50,
    )
    register(
        id='BaxterGolf{}-v0'.format(suffix),
        entry_point='myenvs.baxter.golf:BaxterGolfEnv',
        kwargs=kwargs,
        max_episode_steps=50,
    )
    register(
        id='FetchPnPInAir{}-v1'.format(suffix),
        entry_point='myenvs.fetch.pick_and_place_hard:FetchPickAndPlaceEnv',
        kwargs=kwargs,
        max_episode_steps=50,
    )

    register(
        id='FetchPushMiddleGap{}-v1'.format(suffix),
        entry_point = 'myenvs.fetch.push_wall_obstacle:FetchPushWallObstacle_v1',
        kwargs=kwargs,
        max_episode_steps=50,
    )
    register(
        id='FetchPushWallObstacle{}-v2'.format(suffix),
        entry_point='myenvs.fetch.push_wall_obstacle:FetchPushWallObstacleEnv_v2',
        kwargs=kwargs,
        max_episode_steps=100,
    )
    register(
        id='FetchPushWallObstacle{}-v3'.format(suffix),
        entry_point='myenvs.fetch.push_wall_obstacle:FetchPushWallObstacleEnv_v3',
        kwargs=kwargs,
        max_episode_steps=100,
    )
    register(
        id='FetchPnPObstacle{}-v1'.format(suffix),
        entry_point='myenvs.fetch.pick_and_place_obstacle:FetchPnPObstacle_v1',
        kwargs=kwargs,
        max_episode_steps=50,
    )
    register(
        id='FetchPushWallObstacle{}-v4'.format(suffix),
        entry_point='myenvs.fetch.push_wall_obstacle:FetchPushWallObstacleEnv_v4',
        kwargs=kwargs,
        max_episode_steps=50,
    )
    register(
        id='FetchPushWallObstacle{}-v5'.format(suffix),
        entry_point='myenvs.fetch.push_wall_obstacle:FetchPushWallObstacleEnv_v5',
        kwargs=kwargs,
        max_episode_steps=50,
    )


    register(
        id='FetchCurling{}-v1'.format(suffix),
        entry_point='myenvs.fetch.curling:FetchCurlingEnv',
        kwargs=kwargs,
        max_episode_steps=50,
    )

    register(
        id='FetchPickAndSort{}-v1'.format(suffix),
        entry_point='myenvs.fetch.pick_and_sort:FetchPickAndSortEnv',
        kwargs=kwargs,
        max_episode_steps=100,
    )

    register(
        id='FetchPickObstacle{}-v1'.format(suffix),
        entry_point='myenvs.fetch.pick_obstacle:FetchPickObstacleEnv',
        kwargs=kwargs,
        max_episode_steps=100,
    )

    register(
        id='FetchPushLabyrinth{}-v1'.format(suffix),
        entry_point='myenvs.fetch.push_labyrinth:FetchPushLabyrinthEnv',
        kwargs=kwargs,
        max_episode_steps=100,
    )

    register(
        id='myUR5GripperFind{}-v1'.format(suffix),
        entry_point='myenvs.fetch.myUR5Gripper.find:myUR5GripperFindEnv',
        kwargs=kwargs,
        max_episode_steps=50,
    )
    register(
        id='myUR5GripperFall{}-v1'.format(suffix),
        entry_point='myenvs.fetch.myUR5Gripper.fall:myUR5GripperFallEnv',
        kwargs=kwargs,
        max_episode_steps=100,
    )


    register(
        id='myUR5Poke{}-v1'.format(suffix),
        entry_point='myenvs.fetch.myUR5.poke:myUR5PokeEnv',
        kwargs=kwargs,
        max_episode_steps=50,
    )


    register(
        id='myUR5Find{}-v1'.format(suffix),
        entry_point='myenvs.fetch.myUR5.find:myUR5FindEnv',
        kwargs=kwargs,
        max_episode_steps=50,
    )

    register(
        id='FetchPushObstacle{}-v1'.format(suffix),
        entry_point='myenvs.fetch.push_new:FetchPushNewEnv',
        kwargs=kwargs,
        max_episode_steps=50,
    )

    register(
        id='FetchSlideNew{}-v1'.format(suffix),
        entry_point='myenvs.fetch.slide_new:FetchSlideNewEnv',
        kwargs=kwargs,
        max_episode_steps=50,
    )

    

    register(
        id='FetchPushDoubleObstacle{}-v1'.format(suffix),
        entry_point='myenvs.fetch.push_moving_double_obstacle:FetchPushMovingDoubleObstacleEnv',
        kwargs=kwargs,
        max_episode_steps=100, )
    # register(
    #     id='UR5PickAndPlace{}-v1'.format(suffix),
    #     entry_point='myenvs.robosuite.UR5e:UR5eLift',
    #     kwargs=kwargs,
    #     max_episode_steps=50,
    # )

    register(
        id='DragRope{}-v0'.format(suffix),
        entry_point='myenvs.ravens:DragRopeEnv',
        kwargs=kwargs,
        max_episode_steps=50,
    )

    register(
        id='SweepPile{}-v0'.format(suffix),
        entry_point='myenvs.ravens:SweepPileEnv',
        kwargs=kwargs,
        max_episode_steps=50,
    )

    register(
        id='MsPacman{}-v0'.format(suffix),
        entry_point='myenvs.atari.mspacman:MsPacman',
        kwargs=kwargs,
        max_episode_steps=26,
    )

    _rope_kwargs = {
        'observation_mode': 'key_point',
        'action_mode': 'picker',
        'num_picker': 2,
        'render': True,
        'headless': True,
        'horizon': 50,
        'action_repeat': 8,
        'render_mode': 'cloth',
        'num_variations': 1,
        'use_cached_states': False,
        'deterministic': False,
        'save_cached_states': False,
    }
    _rope_kwargs.update(kwargs)
    register(
        id='RopeConfiguration{}-v0'.format(suffix),
        entry_point='myenvs.softgymenvs:RopeConfigurationEnv',
        kwargs=_rope_kwargs,
        max_episode_steps=50,
    )
    register(
        id='AntMaze{}-v1'.format(suffix),
        entry_point='myenvs.maze.ant_maze:AntMazeEnv',
        kwargs= {},
        max_episode_steps=500,
    )