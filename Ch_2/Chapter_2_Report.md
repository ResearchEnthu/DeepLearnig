# Chapter 2 Report

## Project Overview
This report summarizes the work in `Ch_2/chpater_2..ipynb`, which explores pretrained image classification models from `torchvision` and demonstrates inference using ResNet.

## Objectives
- Explore available pretrained models in `torchvision.models`.
- Instantiate and test AlexNet and ResNet101.
- Preprocess an input image, run model inference, and inspect output predictions.

## Methodology
1. Imported `torchvision.models` and examined available architectures with `dir(models)`.
2. Created an `AlexNet` instance and executed a forward pass with a random input tensor shaped `(1, 3, 224, 224)`.
3. Loaded `ResNet101` with pretrained ImageNet weights.
4. Built a standard preprocessing pipeline using `torchvision.transforms`:
   - Resize to 256 pixels,
   - Center crop to 224x224,
   - Convert to tensor,
   - Normalize with ImageNet mean and standard deviation.
5. Loaded a local image from `OneDrive\Pictures\Screenshots 1\Screenshot 2026-06-21 153140.png` and converted it to RGB.
6. Applied preprocessing, wrapped the image in a batch tensor, and evaluated the model in inference mode.
7. Used softmax to obtain class probabilities and selected top predictions.

## Key Findings
- The notebook successfully constructs and runs pretrained models from `torchvision`.
- AlexNet inference on a random tensor produced an output tensor with appropriate shape.
- ResNet101 evaluation mode and image preprocessing pipeline are correctly applied for inference.
- The code demonstrates both direct `models.resnet101(pretrained=True)` usage and `torch.hub.load(...)` as alternative model-loading strategies.

## Observations and Improvements
- The notebook uses inconsistent variable names such as `Resnet`, `resnet`, `netG`, and `prepocess`. Standardizing names would improve readability.
- The image preprocessing pipeline is defined twice with slightly different variable naming. Consolidating preprocessing into one definition would reduce duplication.
- The notebook does not include ImageNet class label mapping, so the output is currently shown as numeric logits or probabilities rather than human-readable labels.
- A simple check for the model output dimension or a `top-5` label lookup would make the inference result more meaningful.

## Recommendations
- Add a `labels.txt` or `imagenet_classes` mapping to convert model output indices to class names.
- Refactor the notebook to use one consistent ResNet model instance and one preprocessing pipeline.
- Add a short conclusion section to summarize the predicted classes and expected behavior.
- Consider adding a second example image to compare inference results.

## Conclusion
The chapter notebook effectively demonstrates loading pretrained CNNs and running inference on a sample image. With a few cleanups and the addition of label mapping, the notebook would be stronger as a reproducible image classification example for Deep Learning study.
