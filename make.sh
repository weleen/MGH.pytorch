#!/bin/bash
# reranking
cd fastreid/utils/extension
sh make.sh
cd ../../..
# rank
cd fastreid/evaluation/rank_cylib
make
cd ../../..