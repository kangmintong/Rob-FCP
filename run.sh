# pretraining
# federated training on MNIST (IID setting)
# python src/federated_main.py --model=cnn --dataset=mnist --gpu=0 --iid=1 --epochs=10
# federated training on MNIST (non-IID setting)
# python src/federated_main.py --model=cnn --dataset=mnist --gpu=2 --iid=0 --epochs=10


# federated conformal prediction on MNIST (IID setting)
# Baseline
#for type in 'coverage' 'efficiency' 'gaussian_noise';
#do
#  for num in 10 20 30 40 50;
#  do
#    python src/fl_conformal.py  --model=cnn --dataset=mnist  --gpu=2 --iid=1 --num_malicious_clients $num --attack_type $type --robust_conformal 0 --vec_dim 100;
#  done
#done
## Our robust conformal prediction
#for type in 'coverage' 'efficiency' 'gaussian_noise';
#do
#  for num in 10 20 30 40 50 ;
#  do
#    python src/fl_conformal.py  --model=cnn --dataset=mnist  --gpu=2 --iid=1 --num_malicious_clients $num --attack_type $type --robust_conformal 1 --vec_dim 100;
#  done
#done

# federated conformal prediction on MNIST (non-IID setting)
# Baseline
#for type in 'copy_attack' ; # 'coverage' 'efficiency' 'gaussian_noise' 'copy_attack';
#do
#  for num in 10 20 30 40 50;
#  do
#    python src/fl_conformal.py  --model=cnn --dataset=mnist  --gpu=2 --iid=0 --num_malicious_clients $num --attack_type $type --robust_conformal 0 --vec_dim 100;
#  done
#done
### Our robust conformal prediction
#for type in 'copy_attack' ; # 'coverage' 'efficiency' 'gaussian_noise' 'copy_attack';
#do
#  for num in 10 20 30 40 50;
#  do
#    python src/fl_conformal.py  --model=cnn --dataset=mnist  --gpu=2 --iid=0 --num_malicious_clients $num --attack_type $type --robust_conformal 1 --vec_dim 100 --scores2vector histogram;
#  done
#done

# cifar
for iidval in 1 0;
do
  for rob in 0 1; # 0 1
  do
    for type in 'efficiency' ; # 'coverage' 'efficiency' 'gaussian_noise';
    do
      for num in 40 ; # 10 20 30 40 50
      do
        python src/fl_conformal.py  --model=resnet --dataset=cifar  --gpu=0 --iid $iidval --num_malicious_clients $num --attack_type $type --robust_conformal $rob --vec_dim 100 --epochs 100 --num_users 100 --frac 0.2 --local_bs 64;
      done
    done
  done
done

# tinyimagenet
#for iidval in 1 0;
#do
#  for rob in 0 1; # 0 1
#  do
#    for type in 'gaussian_noise'; # 'coverage' 'efficiency' 'gaussian_noise';
#    do
#      for num in 40 ; # 10 20 30 40 50
#      do
#        python src/fl_conformal.py  --model=resnet --dataset=tinyimagenet  --gpu=0 --iid $iidval --num_malicious_clients $num --attack_type $type --gaussian_noise_scale 0.02 --robust_conformal $rob --vec_dim 100 --epochs 100 --num_users 100 --frac 0.2 --local_bs 64;
#      done
#    done
#  done
#done
#python src/fl_conformal.py  --model=resnet --dataset=tinyimagenet  --gpu=7 --iid 2 --num_malicious_clients 5 --num_users 6 --attack_type gaussian_noise --robust_conformal 1 --vec_dim 2 --gaussian_noise_scale 1e-5 --epochs 100 --frac 0.2 --local_bs 64;
#python src/fl_conformal.py  --model=resnet --dataset=cifar  --gpu=8 --iid 2 --num_malicious_clients 5 --num_users 6 --attack_type gaussian_noise --gaussian_noise_scale 0.01 --robust_conformal 1 --vec_dim 2 --epochs 100  --frac 0.2 --local_bs 64;