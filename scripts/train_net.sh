#!/usr/bin/env bash
# enviornment for socket
export NCCL_SOCKET_IFNAME=eno1
export GLOO_SOCKET_IFNAME=eno1

CONFIG=$1
GPU=$2
PY_ARGS=${@：3}

python ./tools/train_net.py --config-file $CONFIG \
                            --num-gpus $GPU \
                            $(PY_ARGS)