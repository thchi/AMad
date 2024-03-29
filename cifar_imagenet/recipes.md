# SGD 

```
python cifar.py -a resnet --depth 20 --epochs 164 --schedule 81 122 --gamma 0.1 --wd 1e-4 --checkpoint ./cps/gadam/checkpoints/cifar10/resnet-20-sgd-01 --gpu-id 0 --model_name sgd_01

python cifar.py -a resnet --depth 20 --epochs 164 --schedule 81 122 --gamma 0.1 --wd 1e-4 --checkpoint ./cps/gadam/checkpoints/cifar10/resnet-20-sgd-003 --gpu-id 0 --model_name sgd_003 --lr 0.03

python cifar.py -a resnet --depth 20 --epochs 164 --schedule 81 122 --gamma 0.1 --wd 1e-4 --checkpoint ./cps/gadam/checkpoints/cifar10/resnet-20-sgd-001 --gpu-id 0 --model_name sgd_001 --lr 0.01

python cifar.py -a resnet --depth 20 --epochs 164 --schedule 81 122 --gamma 0.1 --wd 1e-4 --checkpoint ./cps/gadam/checkpoints/cifar10/resnet-20-sgd-0003 --gpu-id 0 --model_name sgd_0003 --lr 0.003
```

# Vanilla Adam

```
python cifar.py -a resnet --depth 20 --epochs 164 --schedule 81 122 --gamma 0.1 --wd 1e-4 --optimizer adamw  --beta1 0.9 --beta2 0.999  --checkpoint /cps/gadam/checkpoints/cifar10/resnet-20-adam-01 --gpu-id 0 --model_name adam_01

python cifar.py -a resnet --depth 20 --epochs 164 --schedule 81 122 --gamma 0.1 --wd 1e-4 --optimizer adamw  --beta1 0.9 --beta2 0.999  --checkpoint /cps/gadam/checkpoints/cifar10/resnet-20-adam-003 --gpu-id 0 --model_name adam_003 --lr 0.03

python cifar.py -a resnet --depth 20 --epochs 164 --schedule 81 122 --gamma 0.1 --wd 1e-4 --optimizer adamw  --beta1 0.9 --beta2 0.999  --checkpoint /cps/gadam/checkpoints/cifar10/resnet-20-adam-001 --gpu-id 0 --model_name adam_001 --lr 0.01

python cifar.py -a resnet --depth 20 --epochs 164 --schedule 81 122 --gamma 0.1 --wd 1e-4 --optimizer adamw  --beta1 0.9 --beta2 0.999  --checkpoint /cps/gadam/checkpoints/cifar10/resnet-20-adam-0003 --gpu-id 0 --model_name adam_0003 --lr 0.003
```

# AMadam experiments

```
! python /content/AMad/cifar_imagenet/cifar.py -a resnet --depth 20 --data cifar10 --epochs 200 --optimizer  amadam --schedule 81 122 --gamma 0.1 --lr 0.01  --checkpoint checkpoints/cifar10/resnet

! python /content/AMad/cifar_imagenet/cifar.py -a resnet --depth 20 --data cifar10 --epochs 200 --optimizer  amadam --schedule 81 122 --gamma 0.1 --lr 0.001  --checkpoint checkpoints/cifar10/resnet

! python /content/AMad/cifar_imagenet/cifar.py -a alexnet --depth 20 --data cifar10 --epochs 200 --optimizer  amadam --schedule 81 122 --gamma 0.1 --lr 0.001  --checkpoint checkpoints/cifar10/alexnet

! python /content/AMad/cifar_imagenet/cifar.py -a vgg_bn --depth 20 --data cifar10 --epochs 200 --optimizer  amadam --schedule 81 122 --gamma 0.1 --lr 0.001  --checkpoint checkpoints/cifar10/vggxnet
 
```
