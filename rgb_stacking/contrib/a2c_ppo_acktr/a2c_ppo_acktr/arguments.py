import argparse
from dataclasses import dataclass
import torch, yaml

@dataclass
class Arg:
    algo = 'a2c'
    lr = 7e-4
    alpha = 0.99
    gamma = 0.99
    eps = 1e-5
    use_gae = False
    gae_lambda = 0.95
    entropy_coef = 0.01
    value_loss_coef = 0.5
    max_grad_norm = 0.5
    seed = 1
    cuda_deterministic = False
    cuda = False
    num_processes = 16
    num_steps = 5
    ppo_epoch = 4
    num_mini_batch = 32
    clip_param = 0.2
    log_interval = 10
    save_interval = 100
    num_env_steps = 10e6
    env_name = 'CartPole-v1'
    log_dir = '/tmp/gym/'
    save_dir = './trained_models/'
    recurrent_policy: str = None
    no_cuda = True
    use_proper_time_limits = False
    eval_interval: int = None
    use_linear_lr_decay: bool = False


def get_args(path):
    args = Arg()
    with open(path) as file:
        node = yaml.load(file, yaml.Loader)

    for key in node:
        args.__setattr__(key, node[key])

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    assert args.algo in ['a2c', 'ppo', 'acktr']
    if args.recurrent_policy is not None:
        assert args.algo in ['a2c', 'ppo'], \
            'Recurrent policy is not implemented for ACKTR'

    return args
