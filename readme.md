#  Image Super-Resolution Using Very Deep Residual Channel Attention Networks

An implementation of RCAN described in the paper using tensorflow.
[Image Super-Resolution Using Very Deep Residual Channel Attention Networks](https://arxiv.org/abs/1807.02758)

Published in ECCV 2018, written by Y. Zhang, K. Li, L. Wang, B. Zhong, and Y. Fu

## Requirement
- Python 3.6.5
- Tensorflow 1.13.1
- Pillow 6.0.0
- numpy 1.15.0
- scikit-image 0.15.0

## Datasets
- [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/)
- [Benchmarks](https://cv.snu.ac.kr/research/EDSR/benchmark.tar)
 
## Pre-trained model
- [RCAN_X2](https://drive.google.com/open?id=1SIGdGjMieworUG_z2LLay2ysiHCSyjjF)
- [RCAN_X3](https://drive.google.com/open?id=1Dlel4QYXcMU1zJvYlnRG3FoKvCQrOg-N)
- [RCAN_X4](https://drive.google.com/open?id=1QvQQMsFSuuaAmWPYAqImPK2Z3Ka-g42b)


## Train using your own dataset

```
python main.py --train_GT_path ./GT_path --train_LR_path ./LR_path --test_GT_path ./test_GT_path --test_LR_path ./test_LR_path --test_with_train True --scale 2(or 3, 4, ...) --log_freq 1000
```

- LR image and HR image pair should have same index when they are sorted by name respectively.
- You can refer to the script file (run.sh) in my repository

## Test using benchmarks
1) Download pre-trained model.

- [RCAN_X2](https://drive.google.com/open?id=1SIGdGjMieworUG_z2LLay2ysiHCSyjjF)
- [RCAN_X3](https://drive.google.com/open?id=1Dlel4QYXcMU1zJvYlnRG3FoKvCQrOg-N)
- [RCAN_X4](https://drive.google.com/open?id=1QvQQMsFSuuaAmWPYAqImPK2Z3Ka-g42b)

The other pre-trained models for scale 3, 4 are will be updated soon!

2) Unzip the pre-trained model file

```
tar -cvf model.tar
```

3) Test using benchmarks

```
python main.py --mode test --pre_trained_model ./model/RCAN_X2(or 3, 4) --test_LR_path ./benchmark_LR_path --test_GT_path ./benchmark_GT_path --scale 2(or 3, 4) --self_ensemble False
```
If you want to use self_ensemble, --self\_ensemble option to True

- You can refer to the script file (run.sh) in my repository

## Inference your own images
1) Download pre-trained model. 

- [RCAN_X2](https://drive.google.com/open?id=1SIGdGjMieworUG_z2LLay2ysiHCSyjjF)
- [RCAN_X3](https://drive.google.com/open?id=1Dlel4QYXcMU1zJvYlnRG3FoKvCQrOg-N)
- [RCAN_X4](https://drive.google.com/open?id=1QvQQMsFSuuaAmWPYAqImPK2Z3Ka-g42b)

The other pre-trained models for scale 3, 4 are will be updated soon!

2) Unzip the pre-trained model file

```
tar -cvf model.tar
```

3) Inference your own images

```
python main.py --mode test_only --pre_trained_model ./model/RCAN_X2(or 3, 4) --test_LR_path ./your_own_images --scale 2(or 3, 4) --chop_forward False
```

If your images are too large, OOM error can occur. In that case, --chop\_forward option to True

## Experimental Results
### Qunatitative Results
| Method         | Scale | Set5 			| Set14        | B100 | Urban100 |
|----------------|-------|--------------|--------------|------|----------|
|Bicubic         |X2     |33.66 / 0.9299|30.24 / 0.8688|29.56 / 0.8431|26.88 / 0.8403|
|RDN             |X2     |38.24 / 0.9614|34.01 / 0.9212| 32.34 / 0.9017 | 32.89 / 0.9353 |
|RCAN(paper)     |X2     |38.27 / 0.9614|34.12 / 0.9216|32.41 / 0.9027 | 33.34 / 0.9384 |
|RCAN(my results)|X2     |38.25 / 0.9615|34.07 / 0.9216| 32.36 / 0.9020 | 33.12 / 0.9367 |

| Method         | Scale | Set5 			| Set14        | B100 | Urban100 |
|----------------|-------|--------------|--------------|------|----------|
|Bicubic         |X3     |30.39 / 0.8682|27.55 / 0.7742|27.21 / 0.7385|24.46 / 0.7349|
|RDN             |X3     |34.71 / 0.9296|30.57 / 0.8468| 29.26 / 0.8093 | 28.80 / 0.8653 |
|RCAN(paper)     |X3     |34.74 / 0.9299|30.65 / 0.8482|29.32 / 0.8111 | 29.09 / 0.8702 |
|RCAN(my results)|X3     |34.75 / 0.9302|30.61 / 0.8470| 29.31 / 0.8105 | 29.03 / 0.8693 |

| Method         | Scale | Set5 			| Set14        | B100 | Urban100 |
|----------------|-------|--------------|--------------|------|----------|
|Bicubic         |X4     |28.42 / 0.8104|26.00 / 0.7027|25.96 / 0.6675|23.14 / 0.6577|
|RDN             |X4     |32.47 / 0.8990|28.81 / 0.7871| 27.72 / 0.7419 | 26.61 / 0.8028 |
|RCAN(paper)     |X4     |32.63 / 0.9002|28.87 / 0.7889|27.77 / 0.7436 | 26.82 / 0.8087 |
|RCAN(my results)|X4     |32.56 / 0.8996|28.89 / 0.7891|27.78 / 0.7434 | 26.81 / 0.8079 |

Qualitative results are will be updated soon!

## Comments
If you have any questions or comments on my codes, please email to me. [son1113@snu.ac.kr](mailto:son1113@snu.ac.kr)

## Reference
[1] https://github.com/yulunzhang/RCAN