# iid
#python src/fl_conformal_unknown_num_malicious.py  --model=cnn --dataset=mnist  --gpu=2 --iid=1 --num_malicious_clients 40 --attack_type efficiency --robust_conformal 1 --vec_dim 100 --scores2vector histogram --num_mal_est_round 1;
#python src/fl_conformal_unknown_num_malicious.py  --model=cnn --dataset=mnist  --gpu=2 --iid=1 --num_malicious_clients 20 --attack_type coverage --robust_conformal 1 --vec_dim 100 --scores2vector histogram --num_mal_est_round 1;
#python src/fl_conformal_unknown_num_malicious.py  --model=cnn --dataset=mnist  --gpu=2 --iid=1 --num_malicious_clients 40 --attack_type gaussian_noise --gaussian_noise_scale 1.0 --robust_conformal 1 --vec_dim 100 --scores2vector histogram --num_mal_est_round 1;

# non-iid

#python src/fl_conformal_unknown_num_malicious.py  --model=cnn --dataset=mnist  --gpu=2 --iid=0 --num_malicious_clients 5 --attack_type efficiency --robust_conformal 1 --vec_dim 100 --scores2vector histogram --num_mal_est_round 1;
#python src/fl_conformal_unknown_num_malicious.py  --model=cnn --dataset=mnist  --gpu=2 --iid=0 --num_malicious_clients 20 --attack_type coverage --robust_conformal 1 --vec_dim 100 --scores2vector histogram --num_mal_est_round 1;
#python src/fl_conformal_unknown_num_malicious.py  --model=cnn --dataset=mnist  --gpu=2 --iid=0 --num_malicious_clients 5 --attack_type gaussian_noise --gaussian_noise_scale 5.0 --robust_conformal 1 --vec_dim 100 --scores2vector histogram --num_mal_est_round 1;
#python src/fl_conformal_unknown_num_malicious.py  --model=cnn --dataset=mnist  --gpu=2 --iid=0 --num_malicious_clients 40 --attack_type copy_attack --robust_conformal 1 --vec_dim 100 --scores2vector histogram --num_mal_est_round 1;

# mnist
# iid
#for type in 'coverage' 'efficiency' 'gaussian_noise';
#do
#  for num in 40; # 10 20 30 40;
#  do
#    python src/fl_conformal_unknown_num_malicious.py  --model=cnn --dataset=mnist  --gpu=2 --iid=1 --num_malicious_clients $num --attack_type $type --robust_conformal 1 --vec_dim 100 --scores2vector histogram --num_mal_est_round 1;
#  done
#done
# non-iid
#for type in 'coverage' 'efficiency' 'gaussian_noise';
#do
#  for num in 40; # 10 20 30 40;
#  do
#    python src/fl_conformal_unknown_num_malicious.py  --model=cnn --dataset=mnist  --gpu=2 --iid=0 --num_malicious_clients $num --attack_type $type --robust_conformal 1 --gaussian_noise_scale 5.0 --vec_dim 100 --scores2vector histogram --num_mal_est_round 1;
#  done
#done

# mnist
#for iidval in 1 0 ;
#do
#  for rob in 1; # 0 1
#  do
#    for type in 'coverage' 'efficiency' 'gaussian_noise' ; # 'coverage' 'efficiency' 'gaussian_noise';
#    do
#      for num in 10 ; # 10 20 30 40 50
#      do
#        python src/fl_conformal_unknown_num_malicious.py  --model=cnn --dataset=mnist  --gpu=0 --iid $iidval --num_malicious_clients $num --attack_type $type --robust_conformal $rob --vec_dim 100 --scores2vector histogram --num_mal_est_round 1;
#      done
#    done
#  done
#done

# cifar-10
#for iidval in 1 0 ;
#do
#  for rob in 1; # 0 1
#  do
#    for type in 'coverage' 'efficiency' 'gaussian_noise' ; # 'coverage' 'efficiency' 'gaussian_noise';
#    do
#      for num in 10 ; # 10 20 30 40 50
#      do
#        python src/fl_conformal_unknown_num_malicious.py  --model=resnet --dataset=cifar  --gpu=0 --iid $iidval --num_malicious_clients $num --attack_type $type --robust_conformal $rob --vec_dim 100 --epochs 100 --num_users 100 --frac 0.2 --local_bs 64 --num_mal_est_round 1;
#      done
#    done
#  done
#done

# tinyimagenet
#for iidval in 1 0 ;
#do
#  for rob in 1; # 0 1
#  do
#    for type in 'coverage' 'efficiency' 'gaussian_noise' ; # 'coverage' 'efficiency' 'gaussian_noise';
#    do
#      for num in 10 ; # 10 20 30 40 50
#      do
#        python src/fl_conformal_unknown_num_malicious.py  --model=resnet --dataset=tinyimagenet  --gpu=0 --iid $iidval --num_malicious_clients $num --attack_type $type --gaussian_noise_scale 0.02 --robust_conformal $rob --vec_dim 100 --epochs 100 --num_users 100 --frac 0.2 --local_bs 64 --num_mal_est_round 1;
#      done
#    done
#  done
#done

# only estimate malicious client number
for iidval in 1;
do
  for rob in 1; # 0 1
  do
    for type in 'coverage'; # 'coverage' 'efficiency' 'gaussian_noise';
    do
      for num in 0; # 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 32 34 36 38 40 42 44 46 48 50; # 10 20 30 40 50
      do
        python src/fl_conformal_unknown_num_malicious.py --num_est_only 1  --model=resnet --dataset=tinyimagenet  --gpu=0 --iid $iidval --num_malicious_clients $num --attack_type $type --gaussian_noise_scale 0.02 --robust_conformal $rob --vec_dim 100 --epochs 100 --num_users 100 --frac 0.2 --local_bs 64 --num_mal_est_round 1;
      done
    done
  done
done

#for iidval in 1;
#do
#  for rob in 1; # 0 1
#  do
#    for type in 'coverage'; # 'coverage' 'efficiency' 'gaussian_noise';
#    do
#      for num in 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 32 34 36 38 40 42 44 46 48 50 ; # 10 20 30 40 50
#      do
#        python src/fl_conformal_unknown_num_malicious.py  --num_est_only 1 --model=resnet --dataset=cifar  --gpu=1 --iid $iidval --num_malicious_clients $num --attack_type $type --robust_conformal $rob --vec_dim 100 --epochs 100 --num_users 100 --frac 0.2 --local_bs 64 --num_mal_est_round 1;
#      done
#    done
#  done
#done

#for iidval in 1;
#do
#  for rob in 1; # 0 1
#  do
#    for type in 'coverage'; # 'coverage' 'efficiency' 'gaussian_noise';
#    do
#      for num in 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 32 34 36 38 40 42 44 46 48 50 ; # 10 20 30 40 50
#      do
#         python src/fl_conformal_unknown_num_malicious.py --num_est_only 1  --model=cnn --dataset=mnist  --gpu=0 --iid $iidval --num_malicious_clients $num --attack_type $type --robust_conformal $rob --vec_dim 100 --scores2vector histogram --num_mal_est_round 1;
#      done
#    done
#  done
#done