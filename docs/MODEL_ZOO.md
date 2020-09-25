# Model Zoo and Leaderboard

## Introduction

This file documents collection of baselines trained with fastreid. All numbers were obtained with 1 NVIDIA P40 GPU.
The software in use were PyTorch 1.4, CUDA 10.1.

In addition to these official baseline models, you can find more models in [projects/](https://github.com/JDAI-CV/fast-reid/tree/master/projects).

Unsupervised and domain adaptive re-ID methods on public benchmarks. To add some papers not included, you could create an issue or a pull request. **Note:** the following results are copied from their original papers.

### Contents

+ [Unsupervised learning (USL) on object re-ID](#unsupervised-learning-on-object-re-id)
  + [Market-1501](#market-1501)
  + [DukeMTMC-reID](#dukemtmc-reid)
  + [MSMT17](#msmt17)
+ [Unsupervised domain adaptation (UDA) on object re-ID](#unsupervised-domain-adaptation-on-object-re-id)
  + [Market-1501 -> DukeMTMC-reID](#market-1501---dukemtmc-reid)
  + [DukeMTMC-reID -> Market-1501](#dukemtmc-reid---market-1501)
  + [Market-1501 -> MSMT17](#market-1501---msmt17)
  + [DukeMTMC-reID -> MSMT17](#dukemtmc-reid---msmt17)
+ [Supervised on object re-ID](#supervised-on-object-reid)
  + [Market-1501](#market-1501)
  + [DukeMTMC-reID](#dukemtmc-reid)
  + [MSMT17](#msmt17)
  + [VeRi](#veri)
  + [VehicleID](#vehicleid)
  + [VERI-Wild](#veri-wild)


### Unsupervised learning on object re-ID

#### Market-1501

| Method | Venue | Code | mAP(%) | R@1(%) | R@5(%) | R@10(%) | Reference |
| ------ | :------: | :----: | :------: | :------: | :-------: | :------: | :------ |
| Direct infer | - | - | 2.2 | 6.7 | 14.9 | 20.1 | - |
| UDA_TP | arXiv'18 | [PyTorch (OpenUnReID)](../tools/UDA_TP) | 34.7 | 58.6 | 74.0 | 78.9 | [[config]](https://drive.google.com/file/d/1qzb9aVND9ueXYkXxBYl-WDFZ7aY6AR7N/view?usp=sharing) [[model]](https://drive.google.com/file/d/1JPiB4TNPmsYw-qBwEQsg44T6sGy6m8F5/view?usp=sharing) |
| strong_baseline | - | [PyTorch (OpenUnReID)](../tools/strong_baseline) | 70.5 | 87.9 | 95.7 | 97.1 | [[config]](https://drive.google.com/file/d/13Fwe6ser_JKPIXVmnJd3KfBhsivP0OMa/view?usp=sharing) [[model]](https://drive.google.com/file/d/1lRMCDfIyji58oodAMJkl6ucPs4Lx6iws/view?usp=sharing) |
| SpCL+ | arXiv'20 | [PyTorch (OpenUnReID)](../tools/SpCL) | 76.0 | 89.5 | 96.2 | 97.5 | [Self-paced Contrastive Learning with Hybrid Memory for Domain Adaptive Object Re-ID](https://arxiv.org/pdf/2006.02713.pdf) |
| MMT+ | ICLR'20 | [PyTorch (OpenUnReID)](../tools/MMT) | 74.3 | 88.1 | 96.0 | 97.5 | [Mutual Mean-Teaching: Pseudo Label Refinery for Unsupervised Domain Adaptation on Person Re-identification](https://openreview.net/pdf?id=rJlnOhVYPS) |
| SpCL | arXiv'20 | [PyTorch](https://github.com/yxgeee/SpCL) | 72.6 | 87.7 | 95.2 | 96.9 | [Self-paced Contrastive Learning with Hybrid Memory for Domain Adaptive Object Re-ID](https://arxiv.org/pdf/2006.02713.pdf) |
| HCT | CVPR'20 | [Empty](https://github.com/zengkaiwei/HCT) | 56.4 | 80.0 | 91.6 | 95.2 | [Hierarchical Clustering with Hard-batch Triplet Loss for Person Re-identification](https://openaccess.thecvf.com/content_CVPR_2020/papers/Zeng_Hierarchical_Clustering_With_Hard-Batch_Triplet_Loss_for_Person_Re-Identification_CVPR_2020_paper.pdf) |
| MMCL | CVPR'20 | [PyTorch](https://github.com/kennethwdk/MLCReID) | 45.5 | 80.3 | 89.4 | 92.3 | [Unsupervised Person Re-Identification via Multi-Label Classification](https://openaccess.thecvf.com/content_CVPR_2020/papers/Wang_Unsupervised_Person_Re-Identification_via_Multi-Label_Classification_CVPR_2020_paper.pdf) |
| SSL | CVPR'20 | [PyTorch (Unofficial)](https://github.com/ryanaleksander/softened-similarity-learning) | 37.8 | 71.7 | 83.8 | 87.4 | [Unsupervised Person Re-identification via Softened Similarity Learning](https://openaccess.thecvf.com/content_CVPR_2020/papers/Lin_Unsupervised_Person_Re-Identification_via_Softened_Similarity_Learning_CVPR_2020_paper.pdf) |
| BUC | AAAI'19 | [PyTorch](https://github.com/vana77/Bottom-up-Clustering-Person-Re-identification) | 38.3 | 66.2 | 79.6 | 84.5 | [A Bottom-up Clustering Approach to Unsupervised Person Re-identification](https://vana77.github.io/vana77.github.io/images/AAAI19.pdf) |

#### DukeMTMC-reID

| Method | Venue | Code | mAP(%) | R@1(%) | R@5(%) | R@10(%) | Reference |
| ------ | :------: | :----: | :------: | :------: | :-------: | :------: | :------ |
| Direct infer | - | - | 2.3 | 7.5 | 14.7 | 18.1 | - |
| UDA_TP | arXiv'18 | [PyTorch (OpenUnReID)](../tools/UDA_TP) | 42.3 | 64.4 | 76.0 | 79.9 | [[config]](https://drive.google.com/file/d/1GOrQBdYINXK-RQ8OANuVpBpYly8aYzOs/view?usp=sharing) [[model]](https://drive.google.com/file/d/1N8cALZkOzIEcKdSWkCbG83tQ-ADBwa5E/view?usp=sharing) |
| strong_baseline | - | [PyTorch (OpenUnReID)](../tools/strong_baseline) | 54.7 | 72.9 | 83.5 | 87.2 | [[config]](https://drive.google.com/file/d/1fiuKgedqg839vfZMCdzUEnWVfmXE26TT/view?usp=sharing) [[model]](https://drive.google.com/file/d/1BUoshDWxAtY-L5nNYo2zUnOF6PjiqpyN/view?usp=sharing) |
| SpCL+ | arXiv'20 | [PyTorch (OpenUnReID)](../tools/SpCL) | 67.1 | 82.4 | 90.8 | 93.0 | [Self-paced Contrastive Learning with Hybrid Memory for Domain Adaptive Object Re-ID](https://arxiv.org/pdf/2006.02713.pdf) |
| MMT+ | ICLR'20 | [PyTorch (OpenUnReID)](../tools/MMT) | 60.3 | 75.6 | 86.0 | 89.2 | [Mutual Mean-Teaching: Pseudo Label Refinery for Unsupervised Domain Adaptation on Person Re-identification](https://openreview.net/pdf?id=rJlnOhVYPS) |
| SpCL | arXiv'20 | [PyTorch](https://github.com/yxgeee/SpCL) | 65.3 | 81.2 | 90.3 | 92.2 | [Self-paced Contrastive Learning with Hybrid Memory for Domain Adaptive Object Re-ID](https://arxiv.org/pdf/2006.02713.pdf) |
| HCT | CVPR'20 | [Empty](https://github.com/zengkaiwei/HCT) | 50.7 | 69.6 | 83.4 | 87.4 | [Hierarchical Clustering with Hard-batch Triplet Loss for Person Re-identification](https://openaccess.thecvf.com/content_CVPR_2020/papers/Zeng_Hierarchical_Clustering_With_Hard-Batch_Triplet_Loss_for_Person_Re-Identification_CVPR_2020_paper.pdf) |
| MMCL | CVPR'20 | [PyTorch](https://github.com/kennethwdk/MLCReID) | 40.2 | 65.2 | 75.9 | 80.0 | [Unsupervised Person Re-Identification via Multi-Label Classification](https://openaccess.thecvf.com/content_CVPR_2020/papers/Wang_Unsupervised_Person_Re-Identification_via_Multi-Label_Classification_CVPR_2020_paper.pdf) |
| SSL | CVPR'20 | [PyTorch (Unofficial)](https://github.com/ryanaleksander/softened-similarity-learning) | 28.6 | 52.5 | 63.5 | 68.9 | [Unsupervised Person Re-identification via Softened Similarity Learning](https://openaccess.thecvf.com/content_CVPR_2020/papers/Lin_Unsupervised_Person_Re-Identification_via_Softened_Similarity_Learning_CVPR_2020_paper.pdf) |
| BUC | AAAI'19 | [PyTorch](https://github.com/vana77/Bottom-up-Clustering-Person-Re-identification) | 27.5 | 47.4 | 62.6 | 68.4 | [A Bottom-up Clustering Approach to Unsupervised Person Re-identification](https://vana77.github.io/vana77.github.io/images/AAAI19.pdf) |

#### MSMT17

| Method | Venue | Code | mAP(%) | R@1(%) | R@5(%) | R@10(%) | Reference |
| ------ | :------: | :----: | :------: | :------: | :-------: | :------: | :------ |
| MMCL | CVPR'20 | [PyTorch](https://github.com/kennethwdk/MLCReID) | 11.2 | 35.4 | 44.8 | 49.8 | [Unsupervised Person Re-Identification via Multi-Label Classification](https://openaccess.thecvf.com/content_CVPR_2020/papers/Wang_Unsupervised_Person_Re-Identification_via_Multi-Label_Classification_CVPR_2020_paper.pdf) |

### Unsupervised domain adaptation on object re-ID

#### Market-1501 -> DukeMTMC-reID

| Method | Venue | Code | mAP(%) | R@1(%) | R@5(%) | R@10(%) | Reference |
| ------ | :------: | :----: | :------: | :------: | :-------: | :------: | :------ |
| Direct infer | - | - | 28.1 | 49.3 | 64.3 | 69.7 | [[config]](https://drive.google.com/file/d/1FOuW_Hwl2ASPx0iXeDNxZ1R9MwFBr3gx/view?usp=sharing) [[model]](https://drive.google.com/file/d/13dkhrjz-VIH3jCjIep185MLZxFSD_F7R/view?usp=sharing) |
| UDA_TP | arXiv'18 | [PyTorch (OpenUnReID)](../tools/UDA_TP) | 45.7 | 65.5 | 78.0 | 81.7 | [[config]](https://drive.google.com/file/d/1Dvd-D4lTYJ44SJK0gMpTJ-W8cTgMF0vD/view?usp=sharing) [[model]](https://drive.google.com/file/d/1805D3yqtY3QY8pM83BanLkMLBnBSBgIz/view?usp=sharing) |
| strong_baseline | - | [PyTorch (OpenUnReID)](../tools/strong_baseline) | 60.4 | 75.9 | 86.2 | 89.8 | [[config]](https://drive.google.com/file/d/1-y5o5j6_K037s1BKKlY5IHf-hJ37XEtK/view?usp=sharing) [[model]](https://drive.google.com/file/d/1IVTJkfdlubV_bfH_ipxIEsubraxGbQMI/view?usp=sharing) |
| AWB | arXiv'20 | [Empty]() | 71.0 | 83.4 | 91.7 | 93.8 | [Attentive WaveBlock: Complementarity-enhanced Mutual Networks for Unsupervised Domain Adaptation in Person Re-identification](http://arxiv.org/abs/2006.06525) |
| SpCL+ | arXiv'20 | [PyTorch (OpenUnReID)](../tools/SpCL) | 70.4 | 83.8 | 91.2 | 93.4 | [Self-paced Contrastive Learning with Hybrid Memory for Domain Adaptive Object Re-ID](https://arxiv.org/pdf/2006.02713.pdf) |
| MMT+ | ICLR'20 | [PyTorch (OpenUnReID)](../tools/MMT) | 67.7 | 80.3 | 89.9 | 92.9 | [Mutual Mean-Teaching: Pseudo Label Refinery for Unsupervised Domain Adaptation on Person Re-identification](https://openreview.net/pdf?id=rJlnOhVYPS) |
| SpCL | arXiv'20 | [PyTorch](https://github.com/yxgeee/SpCL) | 68.8 | 82.9 | 90.1 | 92.5 | [Self-paced Contrastive Learning with Hybrid Memory for Domain Adaptive Object Re-ID](https://arxiv.org/pdf/2006.02713.pdf) |
| MEB-Net | ECCV'20 | [PyTorch](https://github.com/YunpengZhai/MEB-Net) | 66.1 | 79.6 | 88.3 | 92.2 | [Multiple Expert Brainstorming for Domain Adaptive Person Re-identification](https://arxiv.org/pdf/2007.01546.pdf) |
| MMT | ICLR'20 | [PyTorch](https://github.com/yxgeee/MMT) | 65.1 | 78.0 | 88.8 | 92.5 | [Mutual Mean-Teaching: Pseudo Label Refinery for Unsupervised Domain Adaptation on Person Re-identification](https://openreview.net/pdf?id=rJlnOhVYPS) |
| AD-Cluster | CVPR'20 | [PyTorch](https://github.com/kennethwdk/MLCReID) | 54.1 | 72.6 | 82.5 | 85.5 | [AD-Cluster: Augmented Discriminative Clustering for Domain Adaptive Person Re-identification](https://openaccess.thecvf.com/content_CVPR_2020/papers/Zhai_AD-Cluster_Augmented_Discriminative_Clustering_for_Domain_Adaptive_Person_Re-Identification_CVPR_2020_paper.pdf) |
| SNR | CVPR'20 | - | 58.1 | 76.3 | - | - | [Style Normalization and Restitution for Generalizable Person Re-identification](https://openaccess.thecvf.com/content_CVPR_2020/papers/Jin_Style_Normalization_and_Restitution_for_Generalizable_Person_Re-Identification_CVPR_2020_paper.pdf) |
| MMCL | CVPR'20 | [PyTorch](https://github.com/kennethwdk/MLCReID) | 51.4 | 72.4 | 82.9 | 85.0 | [Unsupervised Person Re-Identification via Multi-Label Classification](https://openaccess.thecvf.com/content_CVPR_2020/papers/Wang_Unsupervised_Person_Re-Identification_via_Multi-Label_Classification_CVPR_2020_paper.pdf) |
| ECN++ | TPAMI'20 | - | 54.4 | 74.0 | 83.7 | 87.4 | [Learning to Adapt Invariance in Memory for Person Re-identification](https://ieeexplore.ieee.org/abstract/document/9018132) |
| UDA_TP | PR'20 | [PyTorch](https://github.com/LcDog/DomainAdaptiveReID) or [OpenUnReID](../tools/UDA_TP) | 49.0 | 68.4 | 80.1 | 83.5 | [Unsupervised Domain Adaptive Re-Identification: Theory and Practice](https://www.sciencedirect.com/science/article/abs/pii/S003132031930473X) |
| SSG | ICCV'19 | [PyTorch](https://github.com/SHI-Labs/Self-Similarity-Grouping) | 53.4 | 73.0 | 80.6 | 83.2 | [Self-similarity Grouping: A Simple Unsupervised Cross Domain Adaptation Approach for Person Re-identification](https://openaccess.thecvf.com/content_ICCV_2019/papers/Fu_Self-Similarity_Grouping_A_Simple_Unsupervised_Cross_Domain_Adaptation_Approach_for_ICCV_2019_paper.pdf) |
| PCB-PAST | ICCV'19 | [PyTorch](https://github.com/zhangxinyu-xyz/PAST-ReID) | 54.3 | 72.4 | - | - | [Self-Training With Progressive Augmentation for Unsupervised Cross-Domain Person Re-Identification](https://openaccess.thecvf.com/content_ICCV_2019/papers/Zhang_Self-Training_With_Progressive_Augmentation_for_Unsupervised_Cross-Domain_Person_Re-Identification_ICCV_2019_paper.pdf) |
| CR-GAN | ICCV'19 | - | 48.6 | 68.9 | 80.2 | 84.7 | [Instance-Guided Context Rendering for Cross-Domain Person Re-Identification](https://openaccess.thecvf.com/content_ICCV_2019/papers/Chen_Instance-Guided_Context_Rendering_for_Cross-Domain_Person_Re-Identification_ICCV_2019_paper.pdf) |
| PDA-Net | ICCV'19 | - | 45.1 | 63.2 | 77.0 | 82.5 | [Cross-Dataset Person Re-Identification via Unsupervised Pose Disentanglement and Adaptation](https://openaccess.thecvf.com/content_ICCV_2019/papers/Li_Cross-Dataset_Person_Re-Identification_via_Unsupervised_Pose_Disentanglement_and_Adaptation_ICCV_2019_paper.pdf) |
| UCDA | ICCV'19 | - | 31.0 | 47.7 | - | - | [A Novel Unsupervised Camera-aware Domain Adaptation Framework for Person Re-identification](https://openaccess.thecvf.com/content_ICCV_2019/papers/Qi_A_Novel_Unsupervised_Camera-Aware_Domain_Adaptation_Framework_for_Person_Re-Identification_ICCV_2019_paper.pdf) |
| ECN | CVPR'19 | [PyTorch](https://github.com/zhunzhong07/ECN) | 40.4 | 63.3 | 75.8 | 80.4 | [Invariance Matters: Exemplar Memory for Domain Adaptive Person Re-identification](https://arxiv.org/pdf/1904.01990.pdf) |
| HHL | ECCV'18 | [PyTorch](https://github.com/zhunzhong07/HHL) | 33.4 | 60.2 | 73.9 | 79.5 | [Generalizing A Person Retrieval Model Hetero- and Homogeneously](https://openaccess.thecvf.com/content_ECCV_2018/papers/Zhun_Zhong_Generalizing_A_Person_ECCV_2018_paper.pdf) |
| SPGAN | CVPR'18 | [PyTorch](https://github.com/Simon4Yan/eSPGAN) or [OpenUnReID](../tools/SPGAN) | 22.3 | 41.1 | 56.6 | 63.0 | [Image-Image Domain Adaptation with Preserved Self-Similarity and Domain-Dissimilarity for Person Re-identification](https://openaccess.thecvf.com/content_cvpr_2018/papers/Deng_Image-Image_Domain_Adaptation_CVPR_2018_paper.pdf) |
| TJ-AIDL | CVPR'18 | - | 23.0 | 44.3 | 59.6 | 65.0 | [Transferable Joint Attribute-Identity Deep Learning for Unsupervised Person Re-Identification](https://openaccess.thecvf.com/content_cvpr_2018/papers/Wang_Transferable_Joint_Attribute-Identity_CVPR_2018_paper.pdf) |
| PUL | TOMM'18 | [PyTorch](https://github.com/hehefan/Unsupervised-Person-Re-identification-Clustering-and-Fine-tuning) | 16.4 | 30.0 | 43.4 | 48.5 | [Unsupervised Person Re-identification: Clustering and Fine-tuning](https://hehefan.github.io/pdfs/unsupervised-person-identification.pdf) |

<!-- | SDA | arXiv'20 | [PyTorch](https://github.com/yxgeee/SDA) | 61.4 | 76.5 | 86.6 | 89.7 | [Structured Domain Adaptation with Online Relation Regularization for Unsupervised Person Re-ID](https://arxiv.org/pdf/2003.06650.pdf) | -->

#### DukeMTMC-reID -> Market-1501

| Method | Venue | Code | mAP(%) | R@1(%) | R@5(%) | R@10(%) | Reference |
| ------ | :------: | :----: | :------: | :------: | :-------: | :------: | :------ |
| Direct infer | - | - | 27.2 | 58.9 | 75.7 | 81.4 | [[config]](https://drive.google.com/file/d/1_gnPfjwf9uTOJyg1VsBzbMNQ-SGuhohP/view?usp=sharing) [[model]](https://drive.google.com/file/d/1MH-eIuWICkkQ8Ka3stXbiTq889yUZjBV/view?usp=sharing) |
| UDA_TP | arXiv'18 | [PyTorch (OpenUnReID)](../tools/UDA_TP) | 52.3 | 76.0 | 87.8 | 91.9 | [[config]](https://drive.google.com/file/d/1NgbBQrM8jbnKJJHQ1WUZ1sPeXvH6luAd/view?usp=sharing) [[model]](https://drive.google.com/file/d/1ciAk7GxnShm8z25hVqarhaG_8fz_tiyX/view?usp=sharing) |
| strong_baseline | - | [PyTorch (OpenUnReID)](../tools/strong_baseline) | 75.6 | 90.9 | 96.6 | 97.8 | [[config]](https://drive.google.com/file/d/1Oe5QQ-NEJy9YsQr7hsMr5CJlZ0XHJS5P/view?usp=sharing) [[model]](https://drive.google.com/file/d/18t9HOCnQzQlgkRkSs8uFaDFYioGRtcLO/view?usp=sharing) |
| AWB | arXiv'20 | [Empty]() | 80.6 | 92.9 | 97.2 | 98.2 | [Attentive WaveBlock: Complementarity-enhanced Mutual Networks for Unsupervised Domain Adaptation in Person Re-identification](http://arxiv.org/abs/2006.06525) |
| SpCL+ | arXiv'20 | [PyTorch (OpenUnReID)](../tools/SpCL) | 78.2 | 90.5 | 96.6 | 97.8 | [Self-paced Contrastive Learning with Hybrid Memory for Domain Adaptive Object Re-ID](https://arxiv.org/pdf/2006.02713.pdf) |
| MMT+ | ICLR'20 | [PyTorch (OpenUnReID)](../tools/MMT) | 80.9 | 92.2 | 97.6 | 98.4 | [Mutual Mean-Teaching: Pseudo Label Refinery for Unsupervised Domain Adaptation on Person Re-identification](https://openreview.net/pdf?id=rJlnOhVYPS) |
| SpCL | arXiv'20 | [PyTorch](https://github.com/yxgeee/SpCL) | 76.7 | 90.3 | 96.2 | 97.7  | [Self-paced Contrastive Learning with Hybrid Memory for Domain Adaptive Object Re-ID](https://arxiv.org/pdf/2006.02713.pdf) |
| MEB-Net | ECCV'20 | [PyTorch](https://github.com/YunpengZhai/MEB-Net) | 76.0 | 89.9 | 96.0 | 97.5 | [Multiple Expert Brainstorming for Domain Adaptive Person Re-identification](https://arxiv.org/pdf/2007.01546.pdf) |
| MMT | ICLR'20 | [PyTorch](https://github.com/yxgeee/MMT) | 71.2 | 87.7 | 94.9 | 96.9  | [Mutual Mean-Teaching: Pseudo Label Refinery for Unsupervised Domain Adaptation on Person Re-identification](https://openreview.net/pdf?id=rJlnOhVYPS) |
| AD-Cluster | CVPR'20 | [PyTorch](https://github.com/kennethwdk/MLCReID) | 68.3 | 86.7 | 94.4 | 96.5 | [AD-Cluster: Augmented Discriminative Clustering for Domain Adaptive Person Re-identification](https://openaccess.thecvf.com/content_CVPR_2020/papers/Zhai_AD-Cluster_Augmented_Discriminative_Clustering_for_Domain_Adaptive_Person_Re-Identification_CVPR_2020_paper.pdf) |
| SNR | CVPR'20 | - | 61.7 | 82.8 | - | - | [Style Normalization and Restitution for Generalizable Person Re-identification](https://openaccess.thecvf.com/content_CVPR_2020/papers/Jin_Style_Normalization_and_Restitution_for_Generalizable_Person_Re-Identification_CVPR_2020_paper.pdf) |
| MMCL | CVPR'20 | [PyTorch](https://github.com/kennethwdk/MLCReID) | 60.4 | 84.4 | 92.8 | 95.0 | [Unsupervised Person Re-Identification via Multi-Label Classification](https://openaccess.thecvf.com/content_CVPR_2020/papers/Wang_Unsupervised_Person_Re-Identification_via_Multi-Label_Classification_CVPR_2020_paper.pdf) |
| ECN++ | TPAMI'20 | - | 63.8 | 84.1 | 92.8 | 95.4 | [Learning to Adapt Invariance in Memory for Person Re-identification](https://ieeexplore.ieee.org/abstract/document/9018132) |
| UDA_TP | PR'20 | [PyTorch](https://github.com/LcDog/DomainAdaptiveReID) or [OpenUnReID](../tools/UDA_TP) | 53.7 | 75.8 | 89.5 | 93.2 | [Unsupervised Domain Adaptive Re-Identification: Theory and Practice](https://www.sciencedirect.com/science/article/abs/pii/S003132031930473X) |
| SSG | ICCV'19 | [PyTorch](https://github.com/SHI-Labs/Self-Similarity-Grouping) | 58.3 | 80.0 | 90.0 | 92.4 | [Self-similarity Grouping: A Simple Unsupervised Cross Domain Adaptation Approach for Person Re-identification](https://openaccess.thecvf.com/content_ICCV_2019/papers/Fu_Self-Similarity_Grouping_A_Simple_Unsupervised_Cross_Domain_Adaptation_Approach_for_ICCV_2019_paper.pdf) |
| PCB-PAST | ICCV'19 | [PyTorch](https://github.com/zhangxinyu-xyz/PAST-ReID) | 54.6 | 78.4 | - | - | [Self-Training With Progressive Augmentation for Unsupervised Cross-Domain Person Re-Identification](https://openaccess.thecvf.com/content_ICCV_2019/papers/Zhang_Self-Training_With_Progressive_Augmentation_for_Unsupervised_Cross-Domain_Person_Re-Identification_ICCV_2019_paper.pdf) |
| CR-GAN | ICCV'19 | - | 54.0 | 77.7 | 89.7 | 92.7 | [Instance-Guided Context Rendering for Cross-Domain Person Re-Identification](https://openaccess.thecvf.com/content_ICCV_2019/papers/Chen_Instance-Guided_Context_Rendering_for_Cross-Domain_Person_Re-Identification_ICCV_2019_paper.pdf) |
| PDA-Net | ICCV'19 | - | 47.6 | 75.2 | 86.3 | 90.2 | [Cross-Dataset Person Re-Identification via Unsupervised Pose Disentanglement and Adaptation](https://openaccess.thecvf.com/content_ICCV_2019/papers/Li_Cross-Dataset_Person_Re-Identification_via_Unsupervised_Pose_Disentanglement_and_Adaptation_ICCV_2019_paper.pdf) |
| UCDA | ICCV'19 | - | 30.9 | 60.4 | - | - | [A Novel Unsupervised Camera-aware Domain Adaptation Framework for Person Re-identification](https://openaccess.thecvf.com/content_ICCV_2019/papers/Qi_A_Novel_Unsupervised_Camera-Aware_Domain_Adaptation_Framework_for_Person_Re-Identification_ICCV_2019_paper.pdf) |
| ECN | CVPR'19 | [PyTorch](https://github.com/zhunzhong07/ECN) | 43.0 | 75.1 | 87.6 | 91.6 | [Invariance Matters: Exemplar Memory for Domain Adaptive Person Re-identification](https://arxiv.org/pdf/1904.01990.pdf) |
| HHL | ECCV'18 | [PyTorch](https://github.com/zhunzhong07/HHL) | 31.4 | 62.2 | 78.8 | 84.05 | [Generalizing A Person Retrieval Model Hetero- and Homogeneously](https://openaccess.thecvf.com/content_ECCV_2018/papers/Zhun_Zhong_Generalizing_A_Person_ECCV_2018_paper.pdf) |
| SPGAN | CVPR'18 | [PyTorch](https://github.com/Simon4Yan/eSPGAN) or [OpenUnReID](../tools/SPGAN) | 22.8 | 51.5 | 70.1 | 76.8 | [Image-Image Domain Adaptation with Preserved Self-Similarity and Domain-Dissimilarity for Person Re-identification](https://openaccess.thecvf.com/content_cvpr_2018/papers/Deng_Image-Image_Domain_Adaptation_CVPR_2018_paper.pdf) |
| TJ-AIDL | CVPR'18 | - | 26.5 | 58.2 | 74.8 | 81.1 | [Transferable Joint Attribute-Identity Deep Learning for Unsupervised Person Re-Identification](https://openaccess.thecvf.com/content_cvpr_2018/papers/Wang_Transferable_Joint_Attribute-Identity_CVPR_2018_paper.pdf) |
| PUL | TOMM'18 | [PyTorch](https://github.com/hehefan/Unsupervised-Person-Re-identification-Clustering-and-Fine-tuning) | 20.5 |  45.5 | 60.7 | 66.7  | [Unsupervised Person Re-identification: Clustering and Fine-tuning](https://hehefan.github.io/pdfs/unsupervised-person-identification.pdf) |

#### Market-1501 -> MSMT17

| Method | Venue | Code | mAP(%) | R@1(%) | R@5(%) | R@10(%) | Reference |
| ------ | :------: | :----: | :------: | :------: | :-------: | :------: | :------ |
| AWB | arXiv'20 | [Empty]() | - | - | - | - | [Attentive WaveBlock: Complementarity-enhanced Mutual Networks for Unsupervised Domain Adaptation in Person Re-identification](http://arxiv.org/abs/2006.06525) |
| SpCL | arXiv'20 | [PyTorch](https://github.com/yxgeee/SpCL) | 25.4 | 51.6 | 64.3 | 69.7 | [Self-paced Contrastive Learning with Hybrid Memory for Domain Adaptive Object Re-ID](https://arxiv.org/pdf/2006.02713.pdf) |
| MMT | ICLR'20 | [PyTorch](https://github.com/yxgeee/MMT) | 22.9 | 49.2 | 63.1 | 68.8 | [Mutual Mean-Teaching: Pseudo Label Refinery for Unsupervised Domain Adaptation on Person Re-identification](https://openreview.net/pdf?id=rJlnOhVYPS) |
| MMCL | CVPR'20 | [PyTorch](https://github.com/kennethwdk/MLCReID) | 15.1 | 40.8 | 51.8 | 56.7 | [Unsupervised Person Re-Identification via Multi-Label Classification](https://openaccess.thecvf.com/content_CVPR_2020/papers/Wang_Unsupervised_Person_Re-Identification_via_Multi-Label_Classification_CVPR_2020_paper.pdf) |
| ECN++ | TPAMI'20 | - | 15.2 | 40.4 | 53.1 | 58.7 | [Learning to Adapt Invariance in Memory for Person Re-identification](https://ieeexplore.ieee.org/abstract/document/9018132) |
| SSG | ICCV'19 | [PyTorch](https://github.com/SHI-Labs/Self-Similarity-Grouping) | 13.2 | 31.6 | - | 49.6 | [Self-similarity Grouping: A Simple Unsupervised Cross Domain Adaptation Approach for Person Re-identification](https://openaccess.thecvf.com/content_ICCV_2019/papers/Fu_Self-Similarity_Grouping_A_Simple_Unsupervised_Cross_Domain_Adaptation_Approach_for_ICCV_2019_paper.pdf) |
| ECN | CVPR'19 | [PyTorch](https://github.com/zhunzhong07/ECN) | 8.5 | 25.3 | 36.3 | 42.1 | [Invariance Matters: Exemplar Memory for Domain Adaptive Person Re-identification](https://arxiv.org/pdf/1904.01990.pdf) |
| PTGAN | CVPR'18 | - | 2.9 | 10.2 | - | 24.4 | [Person Transfer GAN to Bridge Domain Gap for Person Re-Identification](https://openaccess.thecvf.com/content_cvpr_2018/papers/Wei_Person_Transfer_GAN_CVPR_2018_paper.pdf) |

#### DukeMTMC-reID -> MSMT17

| Method | Venue | Code | mAP(%) | R@1(%) | R@5(%) | R@10(%) | Reference |
| ------ | :------: | :----: | :------: | :------: | :-------: | :------: | :------ |
| AWB | arXiv'20 | [Empty]() | - | - | - | - | [Attentive WaveBlock: Complementarity-enhanced Mutual Networks for Unsupervised Domain Adaptation in Person Re-identification](http://arxiv.org/abs/2006.06525) |
| SpCL | arXiv'20 | [PyTorch](https://github.com/yxgeee/SpCL) | 26.5 | 53.1 | 65.8 | 70.5 | [Self-paced Contrastive Learning with Hybrid Memory for Domain Adaptive Object Re-ID](https://arxiv.org/pdf/2006.02713.pdf) |
| MMT | ICLR'20 | [PyTorch](https://github.com/yxgeee/MMT) | 23.3 | 50.1 | 63.9 | 69.8 | [Mutual Mean-Teaching: Pseudo Label Refinery for Unsupervised Domain Adaptation on Person Re-identification](https://openreview.net/pdf?id=rJlnOhVYPS) |
| MMCL | CVPR'20 | [PyTorch](https://github.com/kennethwdk/MLCReID) | 16.2 | 43.6 | 54.3 | 58.9 | [Unsupervised Person Re-Identification via Multi-Label Classification](https://openaccess.thecvf.com/content_CVPR_2020/papers/Wang_Unsupervised_Person_Re-Identification_via_Multi-Label_Classification_CVPR_2020_paper.pdf) |
| ECN++ | TPAMI'20 | - | 16.0 | 42.5 | 55.9 | 61.5 | [Learning to Adapt Invariance in Memory for Person Re-identification](https://ieeexplore.ieee.org/abstract/document/9018132) |
| SSG | ICCV'19 | [PyTorch](https://github.com/SHI-Labs/Self-Similarity-Grouping) | 13.3 | 32.2 | - | 51.2 | [Self-similarity Grouping: A Simple Unsupervised Cross Domain Adaptation Approach for Person Re-identification](https://openaccess.thecvf.com/content_ICCV_2019/papers/Fu_Self-Similarity_Grouping_A_Simple_Unsupervised_Cross_Domain_Adaptation_Approach_for_ICCV_2019_paper.pdf) |
| ECN | CVPR'19 | [PyTorch](https://github.com/zhunzhong07/ECN) | 10.2 | 30.2 | 41.5 | 46.8 | [Invariance Matters: Exemplar Memory for Domain Adaptive Person Re-identification](https://arxiv.org/pdf/1904.01990.pdf) |
| PTGAN | CVPR'18 | - | 3.3 | 11.8 | - | 27.4 | [Person Transfer GAN to Bridge Domain Gap for Person Re-Identification](https://openaccess.thecvf.com/content_cvpr_2018/papers/Wei_Person_Transfer_GAN_CVPR_2018_paper.pdf) |

### Supervised on object reid

#### How to Read the Tables

- The "Name" column contains a link to the config file.
Running `tools/train_net.py` with this config file and 1 GPU will reproduce the model.
- The *model id* column is provided for ease of reference. To check downloaded file integrity, any model on this page contains tis md5 prefix in its file name.
- Training curves and other statistics can be found in `metrics` for each model.

**BoT**:

[Bag of Tricks and A Strong Baseline for Deep Person Re-identification](http://openaccess.thecvf.com/content_CVPRW_2019/papers/TRMTMCT/Luo_Bag_of_Tricks_and_a_Strong_Baseline_for_Deep_Person_CVPRW_2019_paper.pdf). CVPRW2019, Oral.

**AGW**:

[ReID-Survey with a Powerful AGW Baseline](https://github.com/mangye16/ReID-Survey).

**MGN**:

[Learning Discriminative Features with Multiple Granularities for Person Re-Identification](https://arxiv.org/abs/1804.01438v1)

**SBS**:

stronger baseline on top of BoT:

Bag of Freebies(BoF):

1. Circle loss
2. Freeze backbone training
3. Cutout data augmentation & Auto Augmentation
4. Cosine annealing learning rate decay
5. Soft margin triplet loss

Bag of Specials(BoS):

1. Non-local block
2. GeM pooling

#### Market-1501

**BoT**:

| Method | Pretrained | Rank@1 | mAP | mINP | download |
| :---: | :---: | :---: |:---: | :---: | :---: |
| [BoT(R50)](https://github.com/JDAI-CV/fast-reid/blob/master/configs/Market1501/bagtricks_R50.yml) | ImageNet | 94.4% | 86.1% | 59.4% | [model](https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/market_bot_R50.pth) |
| [BoT(R50-ibn)](https://github.com/JDAI-CV/fast-reid/blob/master/configs/Market1501/bagtricks_R50-ibn.yml) | ImageNet | 94.9% | 87.6% | 64.1% | [model](https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/market_bot_R50-ibn.pth) |
| [BoT(S50)](https://github.com/JDAI-CV/fast-reid/blob/master/configs/Market1501/bagtricks_S50.yml) | ImageNet | 95.2% | 88.7% | 66.9% | [model](https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/market_bot_S50.pth) |
| [BoT(R101-ibn)](https://github.com/JDAI-CV/fast-reid/blob/master/configs/Market1501/bagtricks_R101-ibn.yml) | ImageNet| 95.4% | 88.9% | 67.4% | [model](https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/market_bot_R101-ibn.pth) |

**AGW**:

| Method | Pretrained | Rank@1 | mAP | mINP | download |
| :---: | :---: | :---: |:---: | :---: |:---: |
| [AGW(R50)](https://github.com/JDAI-CV/fast-reid/blob/master/configs/Market1501/AGW_R50.yml) | ImageNet | 95.3% | 88.2% | 66.3% | [model](https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/market_agw_R50.pth) |
| [AGW(R50-ibn)](https://github.com/JDAI-CV/fast-reid/blob/master/configs/Market1501/AGW_R50-ibn.yml) | ImageNet | 95.1% | 88.7% | 67.1% | [model](https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/market_agw_R50-ibn.pth) |
| [AGW(S50)](https://github.com/JDAI-CV/fast-reid/blob/master/configs/Market1501/AGW_S50.yml) | ImageNet | 95.3% | 89.3% | 68.5% | [model](https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/market_agw_S50.pth) |
| [AGW(R101-ibn)](https://github.com/JDAI-CV/fast-reid/blob/master/configs/Market1501/AGW_R101-ibn.yml) | ImageNet | 95.5% | 89.5% | 69.5% | [model](https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/market_agw_R101-ibn.pth) |

**SBS**:

| Method | Pretrained | Rank@1 | mAP | mINP | download |
| :---: | :---: | :---: |:---: | :---: |:---:|
| [SBS(R50)](https://github.com/JDAI-CV/fast-reid/blob/master/configs/Market1501/sbs_R50.yml) | ImageNet | 95.4% | 88.2% | 64.8% | [model](https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/market_sbs_R50.pth) |
| [SBS(R50-ibn)](https://github.com/JDAI-CV/fast-reid/blob/master/configs/Market1501/sbs_R50-ibn.yml) | ImageNet | 95.7% | 89.3% | 67.5% | [model](https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/market_sbs_R50-ibn.pth) |
| [SBS(S50)](https://github.com/JDAI-CV/fast-reid/blob/master/configs/Market1501/sbs_S50.yml) | ImageNet | 95.8% | 89.4% | 67.6% | [model](https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/market_sbs_S50.pth) |
| [SBS(R101-ibn)](https://github.com/JDAI-CV/fast-reid/blob/master/configs/Market1501/sbs_R101-ibn.yml) | ImageNet | 96.3% | 90.3% | 70.0% | [model](https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/market_sbs_R101-ibn.pth) |

**MGN**:

| Method | Pretrained | Rank@1 | mAP | mINP | download |
| :---: | :---: | :---: |:---: | :---: | :---:|
| [SBS(R50-ibn)](https://github.com/JDAI-CV/fast-reid/blob/master/configs/Market1501/mgn_R50-ibn.yml) | ImageNet | 95.8% | 89.8% | 67.7% | [model](https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/market_mgn_R50-ibn.pth) |

#### DukeMTMC-reID

**BoT**:

| Method | Pretrained | Rank@1 | mAP | mINP | download |
| :---: | :---: | :---: |:---: | :---: | :---: |
| [BoT(R50)](https://github.com/JDAI-CV/fast-reid/blob/master/configs/DukeMTMC/bagtricks_R50.yml) | ImageNet | 87.2% | 77.0% | 42.1% | [model](https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/duke_bot_R50.pth) |
| [BoT(R50-ibn)](https://github.com/JDAI-CV/fast-reid/blob/master/configs/DukeMTMC/bagtricks_R50-ibn.yml) | ImageNet | 89.3% | 79.6% | 45.2% | [model](https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/duke_bot_R50-ibn.pth) |
| [BoT(S50)](https://github.com/JDAI-CV/fast-reid/blob/master/configs/DukeMTMC/bagtricks_S50.yml) | ImageNet | 90.0% | 80.13% | 45.8% | [model](https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/duke_bot_S50.pth) |
| [BoT(R101-ibn)](https://github.com/JDAI-CV/fast-reid/blob/master/configs/DukeMTMC/bagtricks_R101-ibn.yml) | ImageNet| 91.2% | 81.2% | 47.5% | [model](https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/duke_bot_R101-ibn.pth) |

**AGW**:

| Method | Pretrained | Rank@1 | mAP | mINP | download |
| :---: | :---: | :---: |:---: | :---: | :---:|
| [AGW(R50)](https://github.com/JDAI-CV/fast-reid/blob/master/configs/DukeMTMC/AGW_R50.yml) | ImageNet | 89.0% | 79.9% | 46.1% | [model](https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/duke_agw_R50.pth) |
| [AGW(R50-ibn)](https://github.com/JDAI-CV/fast-reid/blob/master/configs/DukeMTMC/AGW_R50-ibn.yml) | ImageNet | 90.5% | 80.8% | 47.6% | [model](https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/duke_agw_R50-ibn.pth) |
| [AGW(S50)](https://github.com/JDAI-CV/fast-reid/blob/master/configs/DukeMTMC/AGW_S50.yml) | ImageNet | 90.9% | 82.4% | 49.2% | [model](https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/duke_agw_S50.pth) |
| [AGW(R101-ibn)](https://github.com/JDAI-CV/fast-reid/blob/master/configs/DukeMTMC/AGW_R101-ibn.yml) | ImageNet | 91.7% | 82.3% | 50.0% | [model](https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/duke_agw_R101-ibn.pth) |

**SBS**:

| Method | Pretrained | Rank@1 | mAP | mINP | download |
| :---: | :---: | :---: |:---: | :---: | :---:|
| [SBS(R50)](https://github.com/JDAI-CV/fast-reid/blob/master/configs/DukeMTMC/sbs_R50.yml) | ImageNet | 90.3% | 80.3% | 46.5% | [model](https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/duke_sbs_R50.pth) |
| [SBS(R50-ibn)](https://github.com/JDAI-CV/fast-reid/blob/master/configs/DukeMTMC/sbs_R50-ibn.yml) | ImageNet | 90.8% | 81.2% | 47.0% | [model](https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/duke_sbs_R50-ibn.pth) |
| [SBS(S50)](https://github.com/JDAI-CV/fast-reid/blob/master/configs/DukeMTMC/sbs_S50.yml) | ImageNet | 91.0% | 81.4% | 47.6% | [model](https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/duke_sbs_S50.pth) |
| [SBS(R101-ibn)](https://github.com/JDAI-CV/fast-reid/blob/master/configs/DukeMTMC/sbs_R101-ibn.yml) | ImageNet | 91.9% | 83.6% | 51.5% | [model](https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/duke_sbs_R101-ibn.pth) |

**MGN**:

| Method | Pretrained | Rank@1 | mAP | mINP | download |
| :---: | :---: | :---: |:---: | :---: | :---:|
| [SBS(R50-ibn)](https://github.com/JDAI-CV/fast-reid/blob/master/configs/DukeMTMC/mgn_R50-ibn.yml) | ImageNet | 91.1% | 82.0% | 46.8% | [model](https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/duke_mgn_R50-ibn.pth) |

#### MSMT17

**BoT**:

| Method | Pretrained | Rank@1 | mAP | mINP | download |
| :---: | :---: | :---: |:---: | :---: | :---:|
| [BoT(R50)](https://github.com/JDAI-CV/fast-reid/blob/master/configs/MSMT17/bagtricks_R50.yml) | ImageNet | 74.1%  | 50.2% | 10.4% | [model](https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/msmt_bot_R50.pth) |
| [BoT(R50-ibn)](https://github.com/JDAI-CV/fast-reid/blob/master/configs/MSMT17/bagtricks_R50-ibn.yml) | ImageNet | 77.0% | 54.4% | 12.5% | [model](https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/msmt_bot_R50-ibn.pth) |
| [BoT(S50)](https://github.com/JDAI-CV/fast-reid/blob/master/configs/MSMT17/bagtricks_S50.yml) | ImageNet | 80.8% | 59.9% | 16.3% | [model](https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/msmt_bot_S50.pth) |
| [BoT(R101-ibn)](https://github.com/JDAI-CV/fast-reid/blob/master/configs/MSMT17/bagtricks_R101-ibn.yml) | ImageNet| 81.0% | 59.4% | 15.6% | [model](https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/msmt_bot_R101-ibn.pth) |

**AGW**:

| Method | Pretrained | Rank@1 | mAP | mINP | download |
| :---: | :---: | :---: |:---: | :---: | :---:|
| [AGW(R50)](https://github.com/JDAI-CV/fast-reid/blob/master/configs/MSMT17/AGW_R50.yml) | ImageNet | 78.3% | 55.6% | 12.9% | [model](https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/msmt_agw_R50.pth) |
| [AGW(R50-ibn)](https://github.com/JDAI-CV/fast-reid/blob/master/configs/MSMT17/AGW_R50-ibn.yml) | ImageNet | 81.2% | 59.7% | 15.3% | [model](https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/msmt_agw_R50-ibn.pth) |
| [AGW(S50)](https://github.com/JDAI-CV/fast-reid/blob/master/configs/MSMT17/AGW_S50.yml) | ImageNet | 82.6% | 62.6% | 17.7% | [model](https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/msmt_agw_S50.pth) |
| [AGW(R101-ibn)](https://github.com/JDAI-CV/fast-reid/blob/master/configs/MSMT17/AGW_R101-ibn.yml) | ImageNet | 82.0% | 61.4% | 17.3% | [model](https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/msmt_agw_R101-ibn.pth) |

**SBS**:

| Method | Pretrained | Rank@1 | mAP | mINP | download |
| :---: | :---: | :---: |:---: | :---: | :---:|
| [SBS(R50)](https://github.com/JDAI-CV/fast-reid/blob/master/configs/MSMT17/sbs_R50.yml) | ImageNet | 81.8% | 58.4% | 13.9% | [model](https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/msmt_sbs_R50.pth) |
| [SBS(R50-ibn)](https://github.com/JDAI-CV/fast-reid/blob/master/configs/MSMT17/sbs_R50-ibn.yml) | ImageNet | 83.9% | 60.6% | 15.2% | [model](https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/msmt_sbs_R50-ibn.pth) |
| [SBS(S50)](https://github.com/JDAI-CV/fast-reid/blob/master/configs/MSMT17/sbs_S50.yml) | ImageNet | 84.1% | 61.7% | 15.2% | [model](https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/msmt_sbs_S50.pth) |
| [SBS(R101-ibn)](https://github.com/JDAI-CV/fast-reid/blob/master/configs/MSMT17/sbs_R101-ibn.yml) | ImageNet | 84.8% | 62.8% | 16.3% | [model](https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/msmt_sbs_R101-ibn.pth) |

**MGN**:

| Method | Pretrained | Rank@1 | mAP | mINP | download |
| :---: | :---: | :---: |:---: | :---: | :---:|
| [SBS(R50-ibn)](https://github.com/JDAI-CV/fast-reid/blob/master/configs/MSMT17/mgn_R50-ibn.yml) | ImageNet | 85.1% | 65.4% | 18.4% | - |

#### VeRi

**SBS**:

| Method | Pretrained | Rank@1 | mAP | mINP | download |
| :---: | :---: | :---: |:---: | :---: | :---:| 
| [SBS(R50-ibn)](https://github.com/JDAI-CV/fast-reid/blob/master/configs/VeRi/sbs_R50-ibn.yml) | ImageNet | 97.0%  | 81.9% | 46.3% | -|

#### VehicleID

**BoT**:  
Test protocol: 10-fold cross-validation; trained on 4 NVIDIA P40 GPU.

<table>
<thead>
  <tr>
    <th rowspan="3" align="center">Method</th>
    <th rowspan="3" align="center">Pretrained</th>
    <th colspan="6" align="center">Testset size</th>
    <th rowspan="3" align="center">download</th>
  </tr>
  <tr>
    <td colspan="2" align="center">Small</td>
    <td colspan="2" align="center">Medium</td>
    <td colspan="2" align="center">Large</td>
  </tr>
  <tr>
    <td align="center">Rank@1</td>
    <td align="center">Rank@5</td>
    <td align="center">Rank@1</td>
    <td align="center">Rank@5</td>
    <td align="center">Rank@1</td>
    <td align="center">Rank@5</td>
  </tr>
</thead>
<tbody>
  <tr>
    <td nowrap align="center"><a href="https://github.com/JDAI-CV/fast-reid/blob/master/configs/VehicleID/bagtricks_R50-ibn.yml">BoT(R50-ibn)</a></td>
    <td align="center">ImageNet</td>
    <td align="center">86.6%</td>
    <td align="center">97.9%</td>
    <td align="center">82.9%</td>
    <td align="center">96.0%</td>
    <td align="center">80.6%</td>
    <td align="center">93.9%</td>
    <td align="center">-</td>
  </tr>
</tbody>
</table>

#### VERI-Wild

**BoT**:  
Test protocol: Trained on 4 NVIDIA P40 GPU.

<table>
<thead>
  <tr>
    <th rowspan="3" align="center"> Method</th>
    <th rowspan="3" align="center">Pretrained</th>
    <th colspan="9" align="center">Testset size</th>
    <th rowspan="3" align="center">download</th>
  </tr>
  <tr>
    <td colspan="3" align="center">Small</td>
    <td colspan="3" align="center">Medium</td>
    <td colspan="3" align="center">Large</td>
  </tr>
  <tr>
    <td align="center">Rank@1</td>
    <td align="center">mAP</td>
    <td align="center">mINP</td>
    <td align="center">Rank@1</td>
    <td align="center">mAP</td>
    <td align="center">mINP</td>
    <td align="center">Rank@1</td>
    <td align="center">mAP</td>
    <td align="center">mINP</td>
  </tr>
</thead>
<tbody>
  <tr>
    <td nowrap align="center"><a href="https://github.com/JDAI-CV/fast-reid/blob/master/configs/VERIWild/bagtricks_R50-ibn.yml">BoT(R50-ibn)</a></td>
    <td align="center">ImageNet</td>
    <td align="center">96.4%</td>
    <td align="center">87.7%</td>
    <td align="center">69.2%</td>
    <td align="center">95.1%</td>
    <td align="center">83.5%</td>
    <td align="center">61.2%</td>
    <td align="center">92.5%</td>
    <td align="center">77.3%</td>
    <td align="center">49.8%</td>
    <td align="center">-</td>
  </tr>
</tbody>
</table>
