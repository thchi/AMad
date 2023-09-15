To execute this command in Google Colab, you can follow these steps:

Open Google Colab: create a new notebook or open an existing one.# CIFAR and IMAGENET

Run the Command: In a code cell in your Colab notebook, you can run the command you provided:

!python /content/AMad/cifar_imagenet/cifar.py -a resnet --depth 20 --data cifar10 --epochs 200 --optimizer amadam --schedule 81 122 --gamma 0.1 --lr 0.0099 --checkpoint checkpoints/cifar10/resnet


N.B. This folder is modified based on the pytorch classification project [original repo](https://github.com/bearpaw/pytorch-classification). For more details about this code base, please refer to the original repo. 
A training [recipe](/cifar_imagenet/recipes.md) is provided for image classification experiments. 

