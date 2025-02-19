#!/bin/bash

node_type="${SLURM_JOB_NODELIST:0:7}"                           # Select 'alvis-'
node_list="${SLURM_JOB_NODELIST:8}"                             # Select node list

RANK=0
HOST=$(hostname)                                                # Select host
MASTER="$node_type${SLURM_JOB_NODELIST:8:2}"                    # Identify master

node_list="${node_list/]/}"                                     # Remove ']'
node_list="${node_list/-/,}"                                    # Substitute '-' with ','
IFS=',' read -ra node_list <<< $node_list                       # Create a list with ',' as separator

# Iterate over the sorted nodes and expand the range if needed
for ((i = 0; i < ${#node_list[@]} - 1; i++)); do
    current=${node_list[i]}
    next=${node_list[i + 1]}
    
    # Add the current node to the expanded list
    expanded_nodes+=("$current")
    
    # Check the difference between the current and next node
    diff=$((next - current))
    if ((diff > 1)); then
        # Add the missing nodes to the expanded list
        for ((j = 1; j < diff; j++)); do
            expanded_nodes+=($((current + j)))
        done
    fi
done
expanded_nodes+=("${node_list[${#node_list[@]}-1]}")            # Add last element

for node in "${expanded_nodes[@]}"; do
    full_node=$node_type$node                                   # Joint 'alvis-' + ID

    # Conditional execution based on rank
    if [[ $MASTER = $HOST ]]; then
        echo "$full_node is the master node (rank $RANK). Performing master-specific tasks..."
        python3 -m torch.distributed.run --nproc_per_node=1 --nnodes=2 --node_rank=0 --rdzv_id=123 --rdzv_backend=c10d --rdzv_endpoint=localhost:12345 src/multiple.py
        break
    elif [[ $full_node = $HOST ]]; then
        echo "$full_node is a worker node (rank $RANK). Performing worker-specific tasks..."
        python3 -m torch.distributed.run --nproc_per_node=1 --nnodes=2 --node_rank=$RANK --rdzv_id=123 --rdzv_backend=c10d --rdzv_endpoint=$MASTER:12345 src/multiple.py
        break
    fi

    ((RANK++))
done




