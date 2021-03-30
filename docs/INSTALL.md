# Installation

## Requirements

- Linux or macOS with python ≥ 3.6
- PyTorch ≥ 1.5
- torchvision that matches the Pytorch installation. You can install them together at [pytorch.org](https://pytorch.org/) to make sure of this.
- fvcore
- [yacs](https://github.com/rbgirshick/yacs)
- pathspec (for reproduction)
- Cython (optional to compile evaluation code)
- [faiss-gpu](https://github.com/facebookresearch/faiss) `pip install faiss-gpu`
- numpy
- tensorboard (needed for visualization): `pip install tensorboard`
- scipy
- gdown (for automatically downloading pre-train model)
- sklearn
- Pillow

# Set up with Conda
```shell script
conda create -n fastreid python=3.7
conda activate fastreid
conda install pytorch==1.6.0 torchvision tensorboard -c pytorch
pip install -r requirements
```

# Install apex
```bash
git clone https://github.com/NVIDIA/apex
pip install -v --no-cache-dir --global-option="--pyprof" --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

# Compile GNN Reranking and Cython for Testing
```bash
cd $ROOT_DIR
bash ./make.sh
```
