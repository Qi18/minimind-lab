#!/usr/bin/env python3
import os

import torch
import torch.distributed as dist


def main() -> None:
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    value = torch.tensor([float(rank + 1)], device="cuda")
    dist.all_reduce(value, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()

    expected = dist.get_world_size() * (dist.get_world_size() + 1) / 2
    assert value.item() == expected, (rank, value.item(), expected)
    print(
        f"rank={rank} local_rank={local_rank} "
        f"gpu={torch.cuda.get_device_name(local_rank)} sum={value.item():.0f}",
        flush=True,
    )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
