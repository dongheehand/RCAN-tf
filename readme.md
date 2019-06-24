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
- [RCAN_X2](https://drive.google.com/open?id=11IJJzKTPCTgAUq33escop9LOBKGz_S7m)

The pre-trained models for scale 3, 4 are will be updated soon!

## Train using your own dataset

```
python main.py --train_GT_path ./GT_path --train_LR_path ./LR_path --test_GT_path ./test_GT_path --test_LR_path ./test_LR_path --test_with_train True --scale 2(or 3, 4, ...) --log_freq 1000
```

- LR image and HR image pair should have same index when they are sorted by name respectively. 

## Test using benchmarks
1) Download pre-trained model. 

- [RCAN_X2](https://drive.google.com/open?id=11IJJzKTPCTgAUq33escop9LOBKGz_S7m)

The other pre-trained models for scale 3, 4 are will be updated soon!

2) Unzip the pre-trained model file

```
tar -cvf model.tar
```

3) Test using benchmarks

```
python main.py --mode test --pre_trained_model ./model/RCAN_X2 --test_LR_path ./benchmark_LR_path --test_GT_path ./benchmark_GT_path --scale 2(or 3, 4, ...) --self_ensemble False
```
If you want to use self_ensemble, --self\_ensemble option to True

## Inference your own images
1) Download pre-trained model. 

- [RCAN_X2](https://drive.google.com/open?id=11IJJzKTPCTgAUq33escop9LOBKGz_S7m)

The other pre-trained models for scale 3, 4 are will be updated soon!

2) Unzip the pre-trained model file

```
tar -cvf model.tar
```

3) Inference your own images

```
python main.py --mode test_only --pre_trained_model ./model/RCAN_X2 --test_LR_path ./your_own_images --scale 2(or 3, 4, ...) --chop_forward False
```

If your images are too large, OOM error can occur. In that case, --chop\_forward option to True

## Experimental Results
For Set5 benchmark dataset, 

|| RCAN (paper)  | RCAN (my results) |
|------| ------------- | ------------- |
|PSNR| 38.27 | 38.25  |
|SSIM| 0.9614  | 0.9615  |

More exprimental results are will be updated soon!

## Comments
If you have any questions or comments on my codes, please email to me. [son1113@snu.ac.kr](mailto:son1113@snu.ac.kr)

## Reference
[1]. https://github.com/yulunzhang/RCAN