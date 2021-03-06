import torch
from mpi4py import MPI

from rgb_stacking import ROOT
from rgb_stacking.utils.mpi_tools import broadcast, mpi_avg, num_procs, proc_id


def setup_pytorch_for_mpi():
    """
    Avoid slowdowns caused by each separate process's PyTorch using
    more than its fair share of CPU resources.
    """
    # print('Proc %d: Reporting original number of Torch threads as %d.'%(proc_id(), torch.get_num_threads()), flush=True)
    if torch.get_num_threads() == 1:
        return
    fair_num_threads = max(int(torch.get_num_threads() / num_procs()), 1)
    torch.set_num_threads(fair_num_threads)
    # print('Proc %d: Reporting new number of Torch threads as %d.'%(proc_id(), torch.get_num_threads()), flush=True)


def mpi_avg_grads(module, comm, num_learners):
    """ Average contents of gradient buffers across MPI processedef learner_group(learner_ranks: List[int]):
    if proc_id() in learner_ranks:
        return MPI.COMM_WORLD.group.Incl(learner_ranks)
    return Nones. """
    if num_learners == 1:
        return

    for p in module.parameters():
        p_grad_numpy = p.grad.cpu().numpy()  # numpy view of tensor data
        avg_p_grad = mpi_avg(p_grad_numpy, comm)
        p.grad.data = torch.from_numpy(avg_p_grad).float().to(p.device)


def learner_group(num_learners):
    rollout_root = proc_id() % num_learners
    rollout_per_learner_group = MPI.COMM_WORLD.Split(rollout_root, proc_id())

    learner_ranks = [r for r in range(num_learners)]
    if proc_id() < num_learners:
        return MPI.COMM_WORLD.Create_group(MPI.COMM_WORLD.group.Incl(learner_ranks)), rollout_per_learner_group

    return None, rollout_per_learner_group


def sync_params(module):
    """ Sync all parameters of module across all MPI processes. """
    if num_procs() == 1:
        return
    for p in module.parameters():
        p_np = p.cpu().data.numpy()
        broadcast(p_np, ROOT)
        p.data = torch.from_numpy(p_np).float().to(p.device)
