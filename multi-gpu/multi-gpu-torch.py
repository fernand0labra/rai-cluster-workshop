'''
MIT License

Copyright (c) [2025] [Fernando Labra Caso]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

import os
import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp

from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel
from cluster.utils.torch import data, CustomDataset, CustomNet

'''
Multi-GPU Pytorch Code Snippet (Data Parallelism)

This code is intended to showcase the usage of the pytorch framework for distributed training
of a model on a single machine with multiple GPUs

References:
https://pytorch-geometric.readthedocs.io/en/latest/tutorial/multi_gpu_vanilla.html
'''


def run(rank: int, world_size: int, data: tuple):  # Each node process executes the run function
    # Define the master node location
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'

    # Initialize distributed group (node n of N)
    dist.init_process_group('nccl', rank=rank, world_size=world_size)

    train_size = int(0.8 * data.__len__())          # Calculate nr of train samples
    train_rank_size = train_size // world_size      # Calculate nr of rank samples

    # Slice data by train_rank_size and initialize dataloader
    train_data = CustomDataset(*data[rank * train_rank_size:(rank+1) * train_rank_size])
    train_loader = DataLoader(
        train_data, 
        batch_size=train_data.__len__()//10, 
        shuffle=True, 
        sampler=None,
        batch_sampler=None, 
        num_workers=4
    )

    if rank == 0:  # Validate data only on master node
        val_data = CustomDataset(*data[train_size:])
        val_loader = DataLoader(
            val_data, 
            batch_size=val_data.__len__()//10, 
            shuffle=True, 
            sampler=None,
            batch_sampler=None, 
            num_workers=4
        )
        
    torch.manual_seed(12345)
    model = CustomNet().to(rank)                                # Instantiate model
    model = DistributedDataParallel(model, device_ids=[rank])   # Initialize data parallelism (w/ replicated model)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Instantiate optimizer

    for epoch in range(1, 11):  # Train network
        model.train()
        for batch in train_loader:
            batch = batch.to(rank)
            optimizer.zero_grad()
            out = model(batch.x)[:batch.batch_size]
            loss = F.mse_loss(out, batch.y[:batch.batch_size])  # Compute gradients and update parameters
            loss.backward()
            optimizer.step()

        dist.barrier()  # Synchronize all processes by communication of gradients

        if rank == 0:   # Output on master node
            print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}')

        if rank == 0:   # Evaluate accuracy on master node
            model.eval()
            count = correct = 0
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(rank)
                    out = model(batch.x)[:batch.batch_size]
                    correct += (out == batch.y[:batch.batch_size]).sum()
                    count += batch.batch_size
            print(f'Validation Accuracy: {correct/count:.4f}')

        dist.barrier()

    dist.destroy_process_group()  # Deinitizalize distributed group


if __name__ == '__main__':
    dataset = data()
    world_size = torch.cuda.device_count()  # As many nodes as GPUs
    
    # Spawn process 'run' on each node, passing a copy of the data and waiting for other processes to finish (join)
    mp.spawn(run, args=(world_size, dataset), nprocs=world_size, join=True)
