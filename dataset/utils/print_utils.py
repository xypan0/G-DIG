from deepspeed.utils import logger, log_dist, instrument_w_nvtx
import torch


def print_rank_0(message):
    """If distributed is initialized, print only on rank 0."""
    log_dist(message, ranks=[0])


def print_model_parameters(model):
    total_trainable_param, total_nontrainable_param = 0, 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            total_trainable_param += param.numel()
            print_rank_0(f"{name},\t{param.data.shape}")
        else:
            total_nontrainable_param += param.numel()
    print_rank_0(f"total_nontrainable_param = {total_nontrainable_param}")
    print_rank_0(f"total_trainable_param = {total_trainable_param}")


def print_model(model):
    print_rank_0(f"\nmodel is = \n\n{model}")
