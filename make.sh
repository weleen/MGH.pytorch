#!/bin/bash
# reranking
pushd fastreid/utils/extension
sh make.sh
popd
# rank
pushd fastreid/evaluation/rank_cylib
make
popd