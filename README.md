# CFC-Net

This project hosts the official implementation for the paper: 

**CFC-Net: A Critical Feature Capturing Network for Arbitrary-Oriented Object Detection in Remote Sensing Images** [[paper](https://ieeexplore.ieee.org/abstract/document/9488629)] 

**Notes**：Our paper has been accepted by **TGRS**. 

## Abstract

In this paper, we discuss the role of discriminative features in object detection, and then propose a Critical Feature Capturing Network (CFC-Net) to improve detection accuracy from three aspects: building powerful feature representation, refining preset anchors, and optimizing label assignment. The proposed framework creates more powerful semantic representations for objects in  remote sensing images and achieves high-performance real-time object detection. Note that our model is a one-stage detector with only one anchor on each location in feature maps, which is equivalent to the anchor-free methods, thus the inference speed is faster.

## Requirements

* torch >= 1.1
* CUDA version >=10.0

### Installation
```
pip install -r requirements.txt
pip install git+git://github.com/lehduong/torch-warmup-lr.git

cd $ROOT/utils
sh make.sh

cd $ROOT/datasets/DOTA_devkit
sudo apt-get install swig
swig -c++ -python polyiou.i
python setup.py build_ext --inplace
```


### Training
1. Move the dataset to the `$ROOT` directory.
2. Generate imageset files for dataset division via:
```
cd $ROOT/datasets
python generate_imageset.py
```
3. Modify the configuration file `hyp.py` and arguments  in `train.py`, then start training:
```
python train.py
```
### Inference
You can use the following command to test a dataset. Note that `weight`, `img_dir`, `dataset`,`hyp` should be modified as appropriate.
```
python demo.py
```

### Evaluation

Different datasets use different test methods. For UCAS-AOD/HRSC2016/VOC/NWPU VHR-10, you need to prepare labels in the appropriate format in advance. Take evaluation on HRSC2016 for example:
```
cd $ROOT/datasets/evaluate
python hrsc2gt.py
```
then you can conduct evaluation:
```
python eval.py
```
Note that :

- the script  needs to be executed **only once**, but testing on different datasets needs to be executed again.
- the imageset file used in `hrsc2gt.py` is generated from `generate_imageset.py`.

## Main Results


| Method  | Dataset  | Backbone   | Input Size | mAP  |
| ------- | -------- | ---------- | ---------- | ---- |
| CFC-Net | HRSC2016 | ResNet-50  | 416 x 416  | 86.3 |
| CFC-Net | HRSC2016 | ResNet-101 | 800 x 800  | 89.7 |
| CFC-Net | UCAS-AOD | ResNet-50  | 416 x 416  | 89.5 |
| CFC-Net | DOTA     | ResNet-101 | 800 x 800  | 73.5 |

## Detections

* Results on HRSC2016: 
the red bounding box and the green denotes preset anchors and detection results, respectively.
![HRSC_results](https://github.com/ming71/CFC-Net/blob/master/outputs/HRSC.jpg)

* Results on DOTA: 

![DOTA_results](https://github.com/ming71/CFC-Net/blob/master/outputs/DOTA.jpg)

## Citation

If you find our work or code useful in your research, please consider citing:

```
@article{ming2021cfc,
    author={Ming, Qi and Miao, Lingjuan and Zhou, Zhiqiang and Dong, Yunpeng},
    journal={IEEE Transactions on Geoscience and Remote Sensing},
    title={CFC-Net: A Critical Feature Capturing Network for Arbitrary-Oriented Object Detection in Remote-Sensing Images},
    year={2021},
    volume={},
    number={},
    pages={1-14},
    doi={10.1109/TGRS.2021.3095186}
}


```

If you have any questions, please contact me via issue or [email](mq_chaser@126.com).