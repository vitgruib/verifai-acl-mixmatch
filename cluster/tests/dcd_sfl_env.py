"""info['finish'] reaches SFL through DCD's full vec-env stack (CarRacing and Bipedal). Headless."""
import os, sys
os.environ['ACL27_SOFT_RENDER'] = '1'
sys.path.insert(0, os.getcwd())
import numpy as np
import torch
from arguments import parser
from envs.box2d import *
from envs.bipedalwalker import *
from util import _make_env, create_parallel_env

CR = ['--env_name=CarRacing-Bezier-Adversarial-v0', '--num_control_points=12', '--use_categorical_adv=True',
      '--reward_shaping=True', '--num_action_repeat=8', '--frame_stack=4', '--grayscale=False',
      '--crop_frame=False', '--normalize_returns=True', '--seed=88']
BW = ['--env_name=BipedalWalker-Adversarial-v0', '--normalize_returns=True', '--seed=88']

def vec_check(argv, steps):
    args = parser.parse_args(argv + ['--num_processes=2'])
    venv, ued_venv = create_parallel_env(args)
    ued_venv.reset_random()
    levels = ued_venv.get_level()
    obs = venv.reset_agent()
    ndone = 0
    for t in range(steps):
        a = torch.tensor(np.stack([venv.action_space.sample() for _ in range(2)]))
        obs, r, done, infos = venv.step_env(a, reset_random=False)
        for i in np.flatnonzero(np.asarray(done, dtype=bool).reshape(-1)):
            assert 'finish' in infos[i], infos[i]
            ndone += 1
    # replaying: the level must not change across episodes when reset_random=False
    assert [str(x) for x in ued_venv.get_level()] == [str(x) for x in levels]
    venv.close()
    return ndone


def main():
    # 1. CarRacing single env: a finished lap reports finish=True through the CarRacing wrappers
    env = _make_env(parser.parse_args(CR))
    u = env.unwrapped
    env.seed(1); env.reset_random()
    o, r, d, info = env.step(np.zeros(3))
    assert info.get('finish') is False, info
    u.tile_visited_count = len(u.track)
    o, r, d, info = env.step(np.zeros(3))
    assert d and info.get('finish') is True, (d, info)
    print('CarRacing wrapper: finish passes, lap end -> done')
    env.close()


    print('CarRacing vec: episodes completed with finish key =', vec_check(CR, 300))
    print('Bipedal vec: episodes completed with finish key =', vec_check(BW, 2500))
    print('SFL env tests OK')


if __name__ == '__main__':
    main()
