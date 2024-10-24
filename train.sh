# CIFAR-10 ResNet non-iid
#CUDA_VISIBLE_DEVICES=4 python src/federated_main.py --dataset cifar --model resnet --iid 0 --beta 0.5 --epochs 100 --num_users 100 --frac 0.2 --local_bs 64 --lr 0.001 --optimizer adam > train_cifar_noniid.txt

# CIFAR-10 ResNet iid
#CUDA_VISIBLE_DEVICES=0 python src/federated_main.py --dataset cifar --model resnet --iid 1 --beta 0.5 --epochs 100 --num_users 100 --frac 0.2 --local_bs 64 --lr 0.001 --optimizer adam > train_cifar_iid.txt

# tinyimagenet ResNet non-iid
#CUDA_VISIBLE_DEVICES=5 python src/federated_main.py --dataset tinyimagenet --model resnet --iid 0 --beta 0.5 --epochs 20 --num_users 100 --frac 0.2 --local_bs 64 --lr 0.001 --optimizer adam > train_imagenet_noniid.txt

# tinyimagenet ResNet iid
#CUDA_VISIBLE_DEVICES=1 python src/federated_main.py --dataset tinyimagenet --model resnet --iid 1 --beta 0.5 --epochs 20 --num_users 100 --frac 0.2 --local_bs 64 --lr 0.001 --optimizer adam > train_imagenet_iid.txt
#> train_imagenet_iid.text
CUDA_VISIBLE_DEVICES=5 python src/baseline_main.py --dataset tinyimagenet --model resnet --iid 1 --beta 0.5 --epochs 1 --num_users 100 --frac 0.2 --local_bs 64 --lr 0.001 --optimizer adam > train_imagenet_iid_baseline_main.txt
#> train_imagenet_iid.text

