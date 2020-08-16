## Model Zoo

For the leaderboard on public benchmarks, please refer to [LEADERBOARD.md](LEADERBOARD.md).

**Note:** all the models below are selected by the performance on validation sets.

### Unsupervised learning (USL) on object re-ID

- `Direct infer` models are directly tested on the re-ID datasets with ImageNet pre-trained weights.

#### Market-1501

<!-- | Method | Backbone | Pre-trained | mAP(%) | top-1(%) | top-5(%) | top-10(%) | Train time | Download |
| ----- | :------: | :---------: | :----: | :------: | :------: | :-------: | :------: | :------: |
| Direct infer | ResNet50 | ImageNet | 2.2 | 6.7 | 14.9 | 20.1 | n/a |
| [UDA_TP](../tools/UDA_TP) | ResNet50 | ImageNet | 34.7 | 58.6 | 74.0 | 78.9 | ~2h | [[config]](https://drive.google.com/file/d/1qzb9aVND9ueXYkXxBYl-WDFZ7aY6AR7N/view?usp=sharing) [[model]](https://drive.google.com/file/d/1JPiB4TNPmsYw-qBwEQsg44T6sGy6m8F5/view?usp=sharing) [[log]](https://drive.google.com/file/d/1ImlMaZCpzriq9ScHDfKW6CLmzuCX8KkA/view?usp=sharing) |
| [strong_baseline](../tools/strong_baseline) | ResNet50 | ImageNet | 70.5 | 87.9 | 95.7 | 97.1 | ~2.5h | [[config]](https://drive.google.com/file/d/13Fwe6ser_JKPIXVmnJd3KfBhsivP0OMa/view?usp=sharing) [[model]](https://drive.google.com/file/d/1lRMCDfIyji58oodAMJkl6ucPs4Lx6iws/view?usp=sharing) [[log]](https://drive.google.com/file/d/1IlwrtkLj7nJd7AXszFKFfASADbxSOBQ4/view?usp=sharing) |
| [MMT](../tools/MMT/) | ResNet50 | ImageNet | 74.3 | 88.1 | 96.0 | 97.5 | ~4.5h | [[config]](https://drive.google.com/file/d/16GNU2qQdnmX9qYaqoy9w_DxU9myXjBo4/view?usp=sharing) [[model]](https://drive.google.com/file/d/1y-cSb_6gyigbRNPcsIT1ixOpeg1A9WDg/view?usp=sharing) [[log]](https://drive.google.com/file/d/1lPNykPY6AgfMtsVrqcG-4IQO8wD--2mp/view?usp=sharing) |
| [SpCL](../tools/SpCL/) | ResNet50 | ImageNet | 76.0 | 89.5 | 96.2 | 97.5 | ~2h | [[config]](https://drive.google.com/file/d/1D4IEJhlqPvd8OZocavg0UZrHqpyr60sR/view?usp=sharing) [[model]](https://drive.google.com/file/d/1zMKSKYwdNsg2qKJEHpwvjuxzKoH0e4uE/view?usp=sharing) [[log]](https://drive.google.com/file/d/1Xn1RjFFJPfmlCPI0MUdT-FyppIdiKIY-/view?usp=sharing) | -->

| Method | Backbone | Pre-trained | mAP(%) | top-1(%) | top-5(%) | top-10(%) | Train time | Download |
| ----- | :------: | :---------: | :----: | :------: | :------: | :-------: | :------: | :------: |
| Direct infer | ResNet50 | ImageNet | 2.2 | 6.7 | 14.9 | 20.1 | n/a |
| [UDA_TP](../tools/UDA_TP) | ResNet50 | ImageNet | 34.7 | 58.6 | 74.0 | 78.9 | ~2h | [[config]](https://drive.google.com/file/d/1qzb9aVND9ueXYkXxBYl-WDFZ7aY6AR7N/view?usp=sharing) [[model]](https://drive.google.com/file/d/1JPiB4TNPmsYw-qBwEQsg44T6sGy6m8F5/view?usp=sharing) |
| [strong_baseline](../tools/strong_baseline) | ResNet50 | ImageNet | 70.5 | 87.9 | 95.7 | 97.1 | ~2.5h | [[config]](https://drive.google.com/file/d/13Fwe6ser_JKPIXVmnJd3KfBhsivP0OMa/view?usp=sharing) [[model]](https://drive.google.com/file/d/1lRMCDfIyji58oodAMJkl6ucPs4Lx6iws/view?usp=sharing) |
| [MMT](../tools/MMT/) | ResNet50 | ImageNet | 74.3 | 88.1 | 96.0 | 97.5 | ~4.5h | [[config]](https://drive.google.com/file/d/16GNU2qQdnmX9qYaqoy9w_DxU9myXjBo4/view?usp=sharing) [[model]](https://drive.google.com/file/d/1y-cSb_6gyigbRNPcsIT1ixOpeg1A9WDg/view?usp=sharing) |
| [SpCL](../tools/SpCL/) | ResNet50 | ImageNet | 76.0 | 89.5 | 96.2 | 97.5 | ~2h | [[config]](https://drive.google.com/file/d/1D4IEJhlqPvd8OZocavg0UZrHqpyr60sR/view?usp=sharing) [[model]](https://drive.google.com/file/d/1zMKSKYwdNsg2qKJEHpwvjuxzKoH0e4uE/view?usp=sharing) |

#### DukeMTMC-reID

<!-- | Method | Backbone | Pre-trained | mAP(%) | top-1(%) | top-5(%) | top-10(%) | Train time | Download |
| ----- | :------: | :---------: | :----: | :------: | :------: | :-------: | :------: | :------: |
| Direct infer | ResNet50 | ImageNet | 2.3 | 7.5 | 14.7 | 18.1 | n/a |
| [UDA_TP](../tools/UDA_TP) | ResNet50 | ImageNet | 42.3 | 64.4 | 76.0 | 79.9 | ~2h | [[config]](https://drive.google.com/file/d/1GOrQBdYINXK-RQ8OANuVpBpYly8aYzOs/view?usp=sharing) [[model]](https://drive.google.com/file/d/1N8cALZkOzIEcKdSWkCbG83tQ-ADBwa5E/view?usp=sharing) [[log]](https://drive.google.com/file/d/1xktR52dIItFpYHtr0A4u8kTfkwFmk29v/view?usp=sharing) |
| [strong_baseline](../tools/strong_baseline) | ResNet50 | ImageNet | 54.7 | 72.9 | 83.5 | 87.2 | ~2.5h | [[config]](https://drive.google.com/file/d/1fiuKgedqg839vfZMCdzUEnWVfmXE26TT/view?usp=sharing) [[model]](https://drive.google.com/file/d/1BUoshDWxAtY-L5nNYo2zUnOF6PjiqpyN/view?usp=sharing) [[log]](https://drive.google.com/file/d/1ofH_LoRXQeUyTArzNFoX6IxFFgyz46Y9/view?usp=sharing) |
| [MMT](../tools/MMT/) | ResNet50 | ImageNet | 60.3 | 75.6 | 86.0 | 89.2 | ~4.5h | [[config]](https://drive.google.com/file/d/1kXKdq-mZ-wiWrgsss5Ny_vmdTSnvLAhH/view?usp=sharing) [[model]](https://drive.google.com/file/d/11qtWjAgGtjCa_G3G1hWLWj0Mpko9N7D3/view?usp=sharing) [[log]](https://drive.google.com/file/d/1LGSSooEeNXOQWueRSJW0Ypw_bnpYQDch/view?usp=sharing) |
| [SpCL](../tools/SpCL/) | ResNet50 | ImageNet | 67.1 | 82.4 | 90.8 | 93.0 | ~2h | [[config]](https://drive.google.com/file/d/1QXrH0apN0QqsgU0Bie8Vk7kXKDfYxFL_/view?usp=sharing) [[model]](https://drive.google.com/file/d/1B5nlhSj8AfTpzSW8bkx-LMj-Loa1ENul/view?usp=sharing) [[log]](https://drive.google.com/file/d/1ezMtn8ZtM80Gck_cCs6ZyJC9a6efJ5xF/view?usp=sharing) | -->

| Method | Backbone | Pre-trained | mAP(%) | top-1(%) | top-5(%) | top-10(%) | Train time | Download |
| ----- | :------: | :---------: | :----: | :------: | :------: | :-------: | :------: | :------: |
| Direct infer | ResNet50 | ImageNet | 2.3 | 7.5 | 14.7 | 18.1 | n/a |
| [UDA_TP](../tools/UDA_TP) | ResNet50 | ImageNet | 42.3 | 64.4 | 76.0 | 79.9 | ~2h | [[config]](https://drive.google.com/file/d/1GOrQBdYINXK-RQ8OANuVpBpYly8aYzOs/view?usp=sharing) [[model]](https://drive.google.com/file/d/1N8cALZkOzIEcKdSWkCbG83tQ-ADBwa5E/view?usp=sharing) |
| [strong_baseline](../tools/strong_baseline) | ResNet50 | ImageNet | 54.7 | 72.9 | 83.5 | 87.2 | ~2.5h | [[config]](https://drive.google.com/file/d/1fiuKgedqg839vfZMCdzUEnWVfmXE26TT/view?usp=sharing) [[model]](https://drive.google.com/file/d/1BUoshDWxAtY-L5nNYo2zUnOF6PjiqpyN/view?usp=sharing) |
| [MMT](../tools/MMT/) | ResNet50 | ImageNet | 60.3 | 75.6 | 86.0 | 89.2 | ~4.5h | [[config]](https://drive.google.com/file/d/1kXKdq-mZ-wiWrgsss5Ny_vmdTSnvLAhH/view?usp=sharing) [[model]](https://drive.google.com/file/d/11qtWjAgGtjCa_G3G1hWLWj0Mpko9N7D3/view?usp=sharing) |
| [SpCL](../tools/SpCL/) | ResNet50 | ImageNet | 67.1 | 82.4 | 90.8 | 93.0 | ~2h | [[config]](https://drive.google.com/file/d/1QXrH0apN0QqsgU0Bie8Vk7kXKDfYxFL_/view?usp=sharing) [[model]](https://drive.google.com/file/d/1B5nlhSj8AfTpzSW8bkx-LMj-Loa1ENul/view?usp=sharing) |


### Unsupervised domain adaptation (UDA) on object re-ID

- `Direct infer` models are trained on the source-domain datasets ([source_pretrain](../tools/source_pretrain)) and directly tested on the target-domain datasets.
- UDA methods (`MMT`, `SpCL`, etc.) starting from ImageNet means that they are trained end-to-end in only one stage without source-domain pre-training.

#### DukeMTMC-reID -> Market-1501

<!-- | Method | Backbone | Pre-trained | mAP(%) | top-1(%) | top-5(%) | top-10(%) | Train time | Download |
| ----- | :------: | :---------: | :----: | :------: | :------: | :-------: | :------: | :------: |
| Direct infer | ResNet50 | DukeMTMC-reID | 27.2 | 58.9 | 75.7 | 81.4 | ~1h | [[config]](https://drive.google.com/file/d/1_gnPfjwf9uTOJyg1VsBzbMNQ-SGuhohP/view?usp=sharing) [[model]](https://drive.google.com/file/d/1MH-eIuWICkkQ8Ka3stXbiTq889yUZjBV/view?usp=sharing) [[log]](https://drive.google.com/file/d/15NUJvltPs_oT_0pyTjaKaEqn4n5hiyJI/view?usp=sharing) |
| [UDA_TP](../tools/UDA_TP) | ResNet50 | DukeMTMC-reID | 52.3 | 76.0 | 87.8 | 91.9 | ~2h | [[config]](https://drive.google.com/file/d/1NgbBQrM8jbnKJJHQ1WUZ1sPeXvH6luAd/view?usp=sharing) [[model]](https://drive.google.com/file/d/1ciAk7GxnShm8z25hVqarhaG_8fz_tiyX/view?usp=sharing) [[log]](https://drive.google.com/file/d/12-U3hmjhz3D3rtUJ-_vsTE5QkQ5LgxfU/view?usp=sharing) |
| [strong_baseline](../tools/strong_baseline) | ResNet50 | ImageNet | 75.6 | 90.9 | 96.6 | 97.8 | ~3h | [[config]](https://drive.google.com/file/d/1Oe5QQ-NEJy9YsQr7hsMr5CJlZ0XHJS5P/view?usp=sharing) [[model]](https://drive.google.com/file/d/18t9HOCnQzQlgkRkSs8uFaDFYioGRtcLO/view?usp=sharing) [[log]](https://drive.google.com/file/d/1kn77MKbCBDviauLDphCS-NnpfBj_gQXd/view?usp=sharing) |
| [MMT](../tools/MMT/) | ResNet50 | ImageNet | 80.9 | 92.2 | 97.6 | 98.4 | ~6h | [[config]](https://drive.google.com/file/d/1iFiOLbrzVQcEtIlFvsDIcDf4FcT9Z60U/view?usp=sharing) [[model]](https://drive.google.com/file/d/1XGOrt1iTHQNuFPebBcNjPrkTEwBXXRr_/view?usp=sharing) [[log]](https://drive.google.com/file/d/1Hwpr3f0X_EMYzkMsiegF7dqX0WsYkMQw/view?usp=sharing) |
| [SpCL](../tools/SpCL/) | ResNet50 | ImageNet | 78.2 | 90.5 | 96.6 | 97.8 | ~3h | [[config]](https://drive.google.com/file/d/1O8XxCJDzpI7VIRR7crh0kkOK8vebmIgj/view?usp=sharing) [[model]](https://drive.google.com/file/d/1LvrHptXgzWspN2jwYtom4L_jUKYHpU_z/view?usp=sharing) [[log]](https://drive.google.com/file/d/1oy45txWnreyWOp2Y-E5mwjIss7raBDAP/view?usp=sharing) | -->

| Method | Backbone | Pre-trained | mAP(%) | top-1(%) | top-5(%) | top-10(%) | Train time | Download |
| ----- | :------: | :---------: | :----: | :------: | :------: | :-------: | :------: | :------: |
| Direct infer | ResNet50 | DukeMTMC | 27.2 | 58.9 | 75.7 | 81.4 | ~1h | [[config]](https://drive.google.com/file/d/1_gnPfjwf9uTOJyg1VsBzbMNQ-SGuhohP/view?usp=sharing) [[model]](https://drive.google.com/file/d/1MH-eIuWICkkQ8Ka3stXbiTq889yUZjBV/view?usp=sharing) |
| [UDA_TP](../tools/UDA_TP) | ResNet50 | DukeMTMC | 52.3 | 76.0 | 87.8 | 91.9 | ~2h | [[config]](https://drive.google.com/file/d/1NgbBQrM8jbnKJJHQ1WUZ1sPeXvH6luAd/view?usp=sharing) [[model]](https://drive.google.com/file/d/1ciAk7GxnShm8z25hVqarhaG_8fz_tiyX/view?usp=sharing) |
| [strong_baseline](../tools/strong_baseline) | ResNet50 | ImageNet | 75.6 | 90.9 | 96.6 | 97.8 | ~3h | [[config]](https://drive.google.com/file/d/1Oe5QQ-NEJy9YsQr7hsMr5CJlZ0XHJS5P/view?usp=sharing) [[model]](https://drive.google.com/file/d/18t9HOCnQzQlgkRkSs8uFaDFYioGRtcLO/view?usp=sharing) |
| [MMT](../tools/MMT/) | ResNet50 | ImageNet | 80.9 | 92.2 | 97.6 | 98.4 | ~6h | [[config]](https://drive.google.com/file/d/1iFiOLbrzVQcEtIlFvsDIcDf4FcT9Z60U/view?usp=sharing) [[model]](https://drive.google.com/file/d/1XGOrt1iTHQNuFPebBcNjPrkTEwBXXRr_/view?usp=sharing) |
| [SpCL](../tools/SpCL/) | ResNet50 | ImageNet | 78.2 | 90.5 | 96.6 | 97.8 | ~3h | [[config]](https://drive.google.com/file/d/1O8XxCJDzpI7VIRR7crh0kkOK8vebmIgj/view?usp=sharing) [[model]](https://drive.google.com/file/d/1LvrHptXgzWspN2jwYtom4L_jUKYHpU_z/view?usp=sharing) |
<!-- | [SDA](../tools/SDA/) | ResNet50 | DukeMTMC-reID | -->

#### Market-1501 -> DukeMTMC-reID

<!-- | Method | Backbone | Pre-trained | mAP(%) | top-1(%) | top-5(%) | top-10(%) | Train time | Download |
| ----- | :------: | :---------: | :----: | :------: | :------: | :-------: | :------: | :------: |
| Direct infer | ResNet50 | Market-1501 | 28.1 | 49.3 | 64.3 | 69.7 | ~1h | [[config]](https://drive.google.com/file/d/1FOuW_Hwl2ASPx0iXeDNxZ1R9MwFBr3gx/view?usp=sharing) [[model]](https://drive.google.com/file/d/13dkhrjz-VIH3jCjIep185MLZxFSD_F7R/view?usp=sharing) [[log]](https://drive.google.com/file/d/1EDT4ymWGzExyxT0uRXIKBXefjyds79qp/view?usp=sharing) |
| [UDA_TP](../tools/UDA_TP) | ResNet50 | Market-1501 | 45.7 | 65.5 | 78.0 | 81.7 | ~2h | [[config]](https://drive.google.com/file/d/1Dvd-D4lTYJ44SJK0gMpTJ-W8cTgMF0vD/view?usp=sharing) [[model]](https://drive.google.com/file/d/1805D3yqtY3QY8pM83BanLkMLBnBSBgIz/view?usp=sharing) [[log]](https://drive.google.com/file/d/1fl_APkPZXtTfYFLoENX9prd_vllIz3b7/view?usp=sharing) |
| [strong_baseline](../tools/strong_baseline) | ResNet50 | ImageNet | 60.4 | 75.9 | 86.2 | 89.8 | ~3h | [[config]](https://drive.google.com/file/d/1-y5o5j6_K037s1BKKlY5IHf-hJ37XEtK/view?usp=sharing) [[model]](https://drive.google.com/file/d/1IVTJkfdlubV_bfH_ipxIEsubraxGbQMI/view?usp=sharing) [[log]](https://drive.google.com/file/d/1dh50GH7HWi7KTEJ8J2mZg3AJ7j3adJEX/view?usp=sharing) |
| [MMT](../tools/MMT/) | ResNet50 | ImageNet | 67.7 | 80.3 | 89.9 | 92.9 | ~6h | [[config]](https://drive.google.com/file/d/1KcRmKH-8VZudb6N-KHj12DhV3ECmdBuM/view?usp=sharing) [[model]](https://drive.google.com/file/d/1tgqTZDLIZQrPS56PF0Yguy6lfNdSAIa9/view?usp=sharing) [[log]](https://drive.google.com/file/d/16ZOpCglyvzctsnbbkWILCNVmZEGlv0LU/view?usp=sharing) |
| [SpCL](../tools/SpCL/) | ResNet50 | ImageNet | 70.4 | 83.8 | 91.2 | 93.4 | ~3h | [[config]](https://drive.google.com/file/d/1ILiId7BF_49kv4dT1pcZE0HQEdeTPXjU/view?usp=sharing) [[model]](https://drive.google.com/file/d/17WQyMnS7PiDy3EpD2RJbk45LVxcRZNi2/view?usp=sharing) [[log]](https://drive.google.com/file/d/19x3a_ZX5XJIEWp3y-eK_2TdNxYwUktdj/view?usp=sharing) | -->

| Method | Backbone | Pre-trained | mAP(%) | top-1(%) | top-5(%) | top-10(%) | Train time | Download |
| ----- | :------: | :---------: | :----: | :------: | :------: | :-------: | :------: | :------: |
| Direct infer | ResNet50 | Market | 28.1 | 49.3 | 64.3 | 69.7 | ~1h | [[config]](https://drive.google.com/file/d/1FOuW_Hwl2ASPx0iXeDNxZ1R9MwFBr3gx/view?usp=sharing) [[model]](https://drive.google.com/file/d/13dkhrjz-VIH3jCjIep185MLZxFSD_F7R/view?usp=sharing) |
| [UDA_TP](../tools/UDA_TP) | ResNet50 | Market | 45.7 | 65.5 | 78.0 | 81.7 | ~2h | [[config]](https://drive.google.com/file/d/1Dvd-D4lTYJ44SJK0gMpTJ-W8cTgMF0vD/view?usp=sharing) [[model]](https://drive.google.com/file/d/1805D3yqtY3QY8pM83BanLkMLBnBSBgIz/view?usp=sharing) |
| [strong_baseline](../tools/strong_baseline) | ResNet50 | ImageNet | 60.4 | 75.9 | 86.2 | 89.8 | ~3h | [[config]](https://drive.google.com/file/d/1-y5o5j6_K037s1BKKlY5IHf-hJ37XEtK/view?usp=sharing) [[model]](https://drive.google.com/file/d/1IVTJkfdlubV_bfH_ipxIEsubraxGbQMI/view?usp=sharing) |
| [MMT](../tools/MMT/) | ResNet50 | ImageNet | 67.7 | 80.3 | 89.9 | 92.9 | ~6h | [[config]](https://drive.google.com/file/d/1KcRmKH-8VZudb6N-KHj12DhV3ECmdBuM/view?usp=sharing) [[model]](https://drive.google.com/file/d/1tgqTZDLIZQrPS56PF0Yguy6lfNdSAIa9/view?usp=sharing) |
| [SpCL](../tools/SpCL/) | ResNet50 | ImageNet | 70.4 | 83.8 | 91.2 | 93.4 | ~3h | [[config]](https://drive.google.com/file/d/1ILiId7BF_49kv4dT1pcZE0HQEdeTPXjU/view?usp=sharing) [[model]](https://drive.google.com/file/d/17WQyMnS7PiDy3EpD2RJbk45LVxcRZNi2/view?usp=sharing) |
<!-- | [SDA](../tools/SDA/) | ResNet50 | Market-1501 | -->


### Supervised on object re-ID
#### FastReID Model Zoo and Baselines

This file documents collection of baselines trained with fastreid. All numbers were obtained with 1 NVIDIA P40 GPU.
The software in use were PyTorch 1.4, CUDA 10.1.

In addition to these official baseline models, you can find more models in [projects/](https://github.com/JDAI-CV/fast-reid/tree/master/projects).

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

### Market1501 Baselines

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

### DukeMTMC Baseline

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

### MSMT17 Baseline

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
| [SBS(S50)](https://github.com/JDAI-CV/fast-reid/blob/master/configs/MSMT17/sbs_S50.yml) | ImageNet | 84.1% | 61.7% | 15.2% | [model](https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/msmt_sbs_S50-ibn.pth) |
| [SBS(R101-ibn)](https://github.com/JDAI-CV/fast-reid/blob/master/configs/MSMT17/sbs_R101-ibn.yml) | ImageNet | 85.1% | 63.3% | 16.6% | - |

**MGN**:

| Method | Pretrained | Rank@1 | mAP | mINP | download |
| :---: | :---: | :---: |:---: | :---: | :---:|
| [SBS(R50-ibn)](https://github.com/JDAI-CV/fast-reid/blob/master/configs/MSMT17/mgn_R50-ibn.yml) | ImageNet | 85.1% | 65.4% | 18.4% | - |

### VeRi Baseline

**SBS**:

| Method | Pretrained | Rank@1 | mAP | mINP | download |
| :---: | :---: | :---: |:---: | :---: | :---:| 
| [SBS(R50-ibn)](https://github.com/JDAI-CV/fast-reid/blob/master/configs/VeRi/sbs_R50-ibn.yml) | ImageNet | 97.0%  | 81.9% | 46.3% | -|

### VehicleID Baseline

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

### VERI-Wild Baseline

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
