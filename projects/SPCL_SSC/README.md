### SPCL
This project is the implementation of **Self-paced Contrastive Learning with Hybrid Memory for Domain Adaptive Object Re-ID**.

The original implementation is in [SpCL](https://github.com/yxgeee/SpCL/).

### Result
We present the re-implementation result on reid datasets.

| Link | Backbone | Source | Target | Rank@1 | Rank@5 | Rank@10 | mAP | mINP | TPR@FPR=1e-4 | 1e-3 | 1e-2 | download |
| :---: | :---: | :---: |:---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| [Direct Transfer](https://github.com/open-mmlab/OpenUnReID/) | ResNet50 | - | Market1501 | 6.7% | 14.9% | 20.1% | 2.2% | - | - | - | - | - |
| SpCL(re-imp) | ResNet50 | - | Market1501 | 87.3% | 95.1% | 96.9% | 72.9% | 33.7% | 8.5% | 33.2% | 83.4% | - |
| [strong_baseline](https://github.com/open-mmlab/OpenUnReID/) | ResNet50 | - | Market1501 | 87.9% | 95.7% | 97.1% | 70.5% | - | - | - | - | - |
| [MMT](https://github.com/open-mmlab/OpenUnReID/) ICLR'2020 | ResNet50 | - | Market1501 | 86.9% | 95.0% | 97.1% | 71.0% | - | - | - | - | - |
| [SpCL](https://github.com/open-mmlab/OpenUnReID/) NIPS'2020 submission | ResNet50 | - | Market1501 | 89.5% | 96.2% | 97.5% | 76.0% | - | - | - | - | - |
| [Direct Transfer](https://github.com/zkcys001/UDAStrongBaseline) | ResNet50 | DukeMTMC | Market1501 | 64.9% | 78.7% | 83.4% | 32.2% | - | - | - | - | - |
| [Direct Transfer](https://github.com/open-mmlab/OpenUnReID/) | ResNet50 | DukeMTMC | Market1501 | 58.9% | 75.7% | 81.4% | 27.2% | - | - | - | - | - |
| [UDA_TP](https://github.com/zkcys001/UDAStrongBaseline) PR'2020 | ResNet50 | DukeMTMC | Market1501 | 76.0% | 87.8% | 91.9% | 52.3% | - | - | - | - | - |
| [strong_baseline](https://github.com/open-mmlab/OpenUnReID/) | ResNet50 | DukeMTMC | Market1501 | 90.9% | 96.6% | 97.8% | 75.6% | - | - | - | - | - |
| [MMT](https://github.com/open-mmlab/OpenUnReID/) ICLR'2020| ResNet50 | DukeMTMC | Market1501 | 92.2% | 97.6% | 98.4% | 80.9% | - | - | - | -  | - |
| [SpCL](https://github.com/open-mmlab/OpenUnReID/) NIPS'2020 submission | ResNet50 | DukeMTMC | Market1501 | 90.5% | 96.6% | 97.8% | 78.2% | - | - | - | - | - |
| [fast-reid baseline](https://github.com/JDAI-CV/fast-reid) | ResNet50 | DukeMTMC | Market1501 | 91.0% | 96.4% | 97.7% | 78.0% | - | - | - | - | - |
| [MLT](https://github.com/MLT-reid/MLT) NIPS'20 submission | ResNet50 | DukeMTMC | Market1501 | 92.8% | 96.8% | 97.9% | 81.5% | - | - | - | - | - |
| - | - | - | - | - | - | - | - | - | - | - | - | - |
| [Direct Transfer](https://github.com/open-mmlab/OpenUnReID/) | ResNet50 | - | DukeMTMC | 7.5% | 14.7% | 18.1% | 2.3% | - | - | - | - | - |
| SpCL(re-imp) | ResNet50 | - | DukeMTMC | 81.3% | 89.4% | 92.6% | 64.7% | 20.4% | 2.2% | 8.6% | 35.6% | - |
| [strong_baseline](https://github.com/open-mmlab/OpenUnReID/) | ResNet50 | - | DukeMTMC | 72.9% | 83.5% | 87.2% | 54.7% | - | - | - | - | - |
| [MMT](https://github.com/open-mmlab/OpenUnReID/) | ResNet50 | - | DukeMTMC | 71.7% | 84.1% | 88.6% | 57.0% | - | - | - | - | - |
| [SpCL](https://github.com/open-mmlab/OpenUnReID/) | ResNet50 | - | DukeMTMC | 82.4% | 90.8% | 93.0% | 67.1% | - | - | - | - | - |
| [Direct Transfer](https://github.com/zkcys001/UDAStrongBaseline) | ResNet50 | Market1501 | DukeMTMC | 51.3% | 65.3% | 71.7% | 34.1% | - | - | - | - | - |
| [Direct Transfer](https://github.com/open-mmlab/OpenUnReID/) | ResNet50 | Market1501 | DukeMTMC | 49.3% | 64.3% | 69.7% | 28.1% | - | - | - | - | - |
| [UDA_TP](https://github.com/open-mmlab/OpenUnReID/) PR'2020 | ResNet50 | Market1501 | DukeMTMC | 65.5% | 78.0% | 81.7% | 45.7% | - | - | - | - | - |
| [strong_baseline](https://github.com/open-mmlab/OpenUnReID/) | ResNet50 | Market1501 | DukeMTMC | 75.9% | 86.2% | 89.8% | 60.4% | - | - | - | - | - |
| [MMT](https://github.com/open-mmlab/OpenUnReID/) ICLR'2020 | ResNet50 | Market1501 | DukeMTMC | 80.3% | 89.9% | 92.9% | 67.7% | - | - | - | - | - |
| [SpCL](https://github.com/open-mmlab/OpenUnReID/) NIPS'2020 submission | ResNet50 | Market1501 | DukeMTMC | 83.8% | 91.2% | 93.4% | 70.4% | - | - | - | - | - |
| [fast-reid baseline](https://github.com/JDAI-CV/fast-reid) | ResNet50 | Market1501 | DukeMTMC | 80.0% | 89.2% | 92.2% | 66.7% | - | - | - | - | - |
| [MLT](https://github.com/MLT-reid/MLT) NIPS'20 submission | ResNet50 | Market1501 | DukeMTMC | 83.9% | 91.5% | 93.2% | 71.2% | - | - | - | - | - |
| - | - | - | - | - | - | - | - | - | - | - | - | - |
| SpCL(re-imp) | ResNet50 | - | MSMT17_v1 | 45.2% | 57.8% | 62.9% | 20.7% | 1.2% | 0.3% | 1.9% | 13.6% | - |
| [Direct Transfer](https://github.com/MLT-reid/MLT) | ResNet50 | Market1501 | MSMT17_v1 | 29.8% | - | - | 10.3% | 9.3% | - | - | - | - |
| [MMT](https://github.com/yxgeee/MMT) ICLR'20 | ResNet50 | Market1501 | MSMT17_v1 | 49.2% | 63.1% | 68.8% | 22.9% | - | - | - | - | - |
| [MLT](https://github.com/MLT-reid/MLT) NIPS'20 submission | ResNet50 | Market1501 | MSMT17_v1 | 56.6% | - | - | 26.5% | - | - | - | - | - |
| [Direct Transfer](https://github.com/MLT-reid/MLT) | ResNet50 | DukeMTMC | MSMT17_v1 | 34.8% | - | - | 12.5% | 0.3% | - | - | - | - |
| [MMT](https://github.com/yxgeee/MMT) ICLR'20 | ResNet50 | DukeMTMC | MSMT17_v1 | 50.1% | 63.9% | 69.8% | 23.3% | - | - | - | - | - |
| [MLT](https://github.com/MLT-reid/MLT) NIPS'20 submission | ResNet50 | DukeMTMC | MSMT17_v1 | 59.5% | - | - | 27.7% | - | - | - | - | - |
| [MMT](https://github.com/yxgeee/MMT) ICLR'20 | IBN-ResNet50 | Market1501 | MSMT17_v1 | 54.4% | 67.6% | 72.9% | 26.6% | - | - | - | - | - |
| [MMT](https://github.com/yxgeee/MMT) ICLR'20 | IBN-ResNet50 | DukeMTMC | MSMT17_v1 | 58.2% | 71.6% | 76.8% | 29.3% | - | - | - | - | - |
| - | - | - | - | - | - | - | - | - | - | - | - | - |
| SpCL(re-imp) | ResNet50 | - | MSMT17_v2 | - | - | - | - | - | - | - | - | - |