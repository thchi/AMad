To execute Amadam in Google Colab with CIFAR-10and CIFAR-100, you can follow these steps:

Open Google Colab: create a new notebook or open an existing one.# CIFAR-10 CIFAR-100

Run the Command: In a code cell in your Colab notebook:

# clone the repo
!git clone -l -s https://github.com/thchi/AMad.git
# run the model
!python /content/AMad/cifar_imagenet/cifar.py -a resnet --depth 20 --data cifar10 --epochs 200 --optimizer amadam --schedule 81 122 --gamma 0.1 --lr 0.001 --checkpoint checkpoints/cifar10/resnet
# optimizers
name of optimizers, choices include ['sgd', 'adam', 'adamw', 'adamax', 'adagrad', 'adadelta', 'radam', 'rmsprop', 'amadam'] --lr: learning rate --eps: epsilon value used for optimizers
# Running time
Training a one round takes 2~2.5 hours for a single optimzer. To run all experiments would take 2.5 hours x 9 optimizers x 5 repeats = 112 hours

N.B. This folder is modified based on the pytorch classification project [original repo](https://github.com/bearpaw/pytorch-classification). For more details about this code base, please refer to the original repo. 
A training [recipe](/cifar_imagenet/recipes.md) is provided for image classification experiments. 

