
# PyTorch Implementation of Stochastic Integrated Explanations


## Introduction
This is the official PyTorch implementation of the Stochastic Integrated Explanations (SIX) method.

We introduce a novel method that enables visualization of predictions made by vision models, as well as visualization of explanations for a specific class.
In this method, we present the concept of stochastic integrated explanations.

## Producing SIX Classification Saliency Maps
Images should be stored in the `data\ILSVRC2012_img_val` directory. 
The information on the images being evaluated and their target class can be found in the `data\pics.txt` file, with each line formatted as follows: `<file_name> target_class_number` (e.g. `ILSVRC2012_val_00002214.JPEG 153`).

To generate saliency maps using our method on CNN, run the following command:
```
python cnn_saliency_map_generator.py
```
And to produce maps for ViT, run:
```
python vit_saliency_map_generator.py
```

By default, saliency maps for CNNs are generated using the Resnet101 network on the last layer. You can change the selected network and layer by modifying the `model_name` and `FEATURE_LAYER_NUMBER` variables in the `cnn_saliency_map_generator.py` class. For ViT, the default is ViT-Base, which can also be configured using the `model_name` variable in the `vit_saliency_map_generator.py` class.

The generated saliency maps will be stored in the `qualitive_results` directory.
### ViT models weight files:
- ViT-B [Link to download](https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth)
- ViT-S [Link to download](https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/vit_small_p16_224-15ec54c9.pth)


## Credits
For comparison, we used the following implementations of code from git repositories:
- https://github.com/jacobgil/pytorch-grad-cam
- https://github.com/pytorch/captum
- https://github.com/PAIR-code/saliency

