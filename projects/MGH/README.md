# 项目
MGH文件夹中包含**MGH: Metadata Guided Hypergraph Modeling for Unsupervised Person Re-identification**的具体实现。

该项目以AAAI2021文章**Camera-aware Proxies for Unsupervised Person Re-Identification**为基础，针对行人重识别的无监督领域自适应问题进行了如下修改：
1. 使用超图对聚类标签进行了修正；
2. 增加了基于memory的listwise loss，用于近似优化mAP。

整体的pipeline为**特征提取->聚类->赋伪标签->训练**的循环执行。

## 说明

### 环境配置
项目根目录为
```
ROOT_DIR
```
环境配置请参考项目根目录下
```
$ROOT_DIR/docs/GETTING_STARTED.md
```
建议使用conda进行项目管理，需要注意的是安装apex建议通过编译的方式，在CUDA版本较高的情况下apex目前没有提供安装包。
在运行make.sh时，编译了Cython文件用于加速测试，也编译了GNN Re-ranking。

### 数据集
本项目在Market1501、DukeMTMC-reid、MSMT17上进行了验证。
数据集配置请参考路径
```
$ROOT_DIR/datasets/README.md
```

### 训练
代码运行入口为projects/MGH/train_net.py
训练脚本为projects/MGH/train.sh，
执行训练脚本
```bash
cd $ROOT_DIR
bash projects/MGH/reproduce.sh
```

### 测试
测试脚本为projects/MGH/test.sh，
执行测试脚本前请下载已经训练好的模型models.zip并解压至$ROOT_DIR/models，文件目录如下
```
.
├── duke
│   └── model_duke.pth
├── market
│   └── model_market.pth
└── msmt17
    └── model_msmt17.pth
```
运行测试脚本
```bash
cd $ROOT_DIR
bash project/MGH/test.sh
```

### 代码结构
本项目仿照fastreid中其他项目，继承fastreid已有类及其函数。
其中fastreid原始文件主要包含
```
fastreid/engine/defaults.py(trainer)
fastreid/engine/hooks.py(hooks)
fastreid/engine/train_loop.py(trainer)
```
在此基础上，我们增加了聚类、loss以及后处理代码，对应文件为
```
cap_labelgenerator.py (聚类)
unified_memory.py (loss)
get_st_matrix.py和cluster.py(后处理)
```
其中cap_labelgenerator.py继承自fastreid/engine/hooks.py中的LabelGeneratorHook，集成了多个聚类方法。
get_st_matrix.py中的spatial_temporal_distribution函数统计了时空分布。
此外，超图的构建在hyperg文件夹中。