import argparse
from dataclasses import dataclass
import torch, yaml
from typing import Dict


@dataclass
class PolicyOption:
    feature_extract: Dict = None
    policy_keys: Dict = None
    value_keys: Dict = None
    image_keys: Dict = None
    fc_size = 256
    act_fn = 'relu'
    rec_type: str = None
    hidden_size = 256
    horizon_length: int = None


@dataclass
class Arg:
    model: PolicyOption = None
    algo: str = 'a2c'
    plr: float = 7e-4
    vlr: float = 1e-4
    alpha: float = 0.99
    gamma: float = 0.99
    eps: float = 1e-5
    use_gae: bool = False
    gae_lambda: float = 0.95
    entropy_coef: float = 0.01
    value_loss_coef: float = 0.5
    max_grad_norm: float = 0.5
    seed: int = 1
    cuda_deterministic: bool = False
    cuda: bool = False
    num_processes: int = 16
    num_steps: int = 5
    ppo_epoch: int = 4
    num_mini_batch: int = 32
    clip_param: float = 0.2
    log_interval: int = 10
    save_interval: int = 100
    num_env_steps: int = 10e6
    env_name = 'CartPole-v1'
    log_dir = '/tmp/gym/'
    save_dir = './trained_models/'
    recurrent_policy: str = None
    no_cuda: bool = True
    use_proper_time_limits: bool = False
    eval_interval: int = None
    use_linear_lr_decay: bool = False


def parse_model(yaml_entry):
    opt = PolicyOption()
    for key in yaml_entry:
        opt.__setattr__(key, yaml_entry[key])
    return opt


def get_args(path):
    args = Arg()
    with open(path) as file:
        node = yaml.load(file, yaml.Loader)

    for key in node:
        if key != 'model':
            args.__setattr__(key, node[key])
        else:
            args.__setattr__(key, parse_model(node[key]))

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    if not args.model.horizon_length:
        args.model.horizon_length  = args.num_steps


    assert args.algo in ['a2c', 'ppo', 'acktr']
    if args.recurrent_policy is not None:
        assert args.algo in ['a2c', 'ppo'], \
            'Recurrent policy is not implemented for ACKTR'

    return args
