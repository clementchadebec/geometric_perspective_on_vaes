#!/bin/bash

################## MNIST ##################

######## training ######## 

#python peers/Regularized_autoencoders-RAE-/train_raes_vaes.py 0
#python peers/Regularized_autoencoders-RAE-/train_raes_vaes.py 3
#python peers/Regularized_autoencoders-RAE-/train_raes_vaes.py 4
#python peers/Regularized_autoencoders-RAE-/train_raes_vaes.py 5
#python peers/Regularized_autoencoders-RAE-/train_raes_vaes.py 6
#python train_vae.py data_folders/mnist/mnist_32x32.npz --model 'vamp'
#python train_vae.py data_folders/mnist/mnist_32x32.npz --model 'vae'
#python train_vae.py data_folders/mnist/mnist_32x32.npz --model 'ae'
#python train_vae.py data_folders/mnist/mnist_32x32.npz --model 'rhvae'


######## generation ########

#python peers/Regularized_autoencoders-RAE-/interpolation_fid_and_viz.py 0
#python peers/Regularized_autoencoders-RAE-/interpolation_fid_and_viz.py 3
#python peers/Regularized_autoencoders-RAE-/interpolation_fid_and_viz.py 4
#python peers/Regularized_autoencoders-RAE-/interpolation_fid_and_viz.py 5
#python peers/Regularized_autoencoders-RAE-/interpolation_fid_and_viz.py 6
#python generate_data.py --model_path trained_vae_models/vamp/mnist/best_model.pt --data_path data_folders/mnist/mnist_32x32.npz --n_samples 10000
#python generate_data.py --model_path trained_vae_models/vae/mnist/best_model.pt --data_path data_folders/mnist/mnist_32x32.npz --n_samples 10000
#python generate_data.py --model_path trained_vae_models/vae/mnist/best_model.pt --data_path data_folders/mnist/mnist_32x32.npz --n_samples 10000 --generation 'gauss'
#python generate_data.py --model_path trained_vae_models/vae/mnist/best_model.pt --data_path data_folders/mnist/mnist_32x32.npz --n_samples 10000 --generation 'gmm'
#python generate_data.py --model_path trained_vae_models/ae/mnist/best_model.pt --data_path data_folders/mnist/mnist_32x32.npz --n_samples 10000
#python generate_data.py --model_path trained_vae_models/ae/mnist/best_model.pt --data_path data_folders/mnist/mnist_32x32.npz --n_samples 10000 --generation 'gauss'
#python generate_data.py --model_path trained_vae_models/ae/mnist/best_model.pt --data_path data_folders/mnist/mnist_32x32.npz --n_samples 10000 --generation 'gmm'
#python generate_data.py --model_path trained_vae_models/rhvae/mnist/best_model.pt --data_path data_folders/mnist/mnist_32x32.npz --n_samples 10000 --generation 'gauss'

######## metric computation ########

#python TTUR/fid.py peers/logs/0/WAE_1/one_gaussian_sampled/ data_folders/mnist/test_folder --gpu '0'
#python TTUR/fid.py peers/logs/3/RAE-GP_1/GMM_10_sampled/ data_folders/mnist/test_folder --gpu '0'
#python TTUR/fid.py peers/logs/4/RAE-L2_1/GMM_10_sampled/ data_folders/mnist/test_folder --gpu '0'
#python TTUR/fid.py peers/logs/5/RAE-SN_1/GMM_10_sampled/ data_folders/mnist/test_folder --gpu '0'
#python TTUR/fid.py peers/logs/6/RAE_1/GMM_10_sampled/ data_folders/mnist/test_folder --gpu '0'
#python TTUR/fid.py generated_data/vamp/mnist/ data_folders/mnist/test_folder/ --gpu '0'
#python TTUR/fid.py generated_data/vae/mnist/manifold_sampling/ data_folders/mnist/test_folder/ --gpu '0'
#python TTUR/fid.py generated_data/vae/mnist/gmm/ data_folders/mnist/test_folder/ --gpu '0'
#python TTUR/fid.py generated_data/vae/mnist/gaussian_prior/ data_folders/mnist/test_folder/ --gpu '0'
#python TTUR/fid.py generated_data/ae/mnist/gmm/ data_folders/mnist/test_folder/ --gpu '0'
#python TTUR/fid.py generated_data/ae/mnist/gaussian_prior/ data_folders/mnist/test_folder/ --gpu '0'
#python TTUR/fid.py generated_data/rhvae/mnist/gaussian_prior/ data_folders/mnist/test_folder/ --gpu '0'

#python precision-recall-distributions/prd_from_image_folders.py --inception_path /tmp/classify_image_graph_def.pb --reference_dir data_folders/mnist/test_folder/ --eval_dirs peers/logs/0/WAE_1/one_gaussian_sampled/
#python precision-recall-distributions/prd_from_image_folders.py --inception_path /tmp/classify_image_graph_def.pb --reference_dir data_folders/mnist/test_folder/ --eval_dirs peers/logs/3/RAE-GP_1/GMM_10_sampled/
#python precision-recall-distributions/prd_from_image_folders.py --inception_path /tmp/classify_image_graph_def.pb --reference_dir data_folders/mnist/test_folder/ --eval_dirs peers/logs/4/RAE-L2_1/GMM_10_sampled/
#python precision-recall-distributions/prd_from_image_folders.py --inception_path /tmp/classify_image_graph_def.pb --reference_dir data_folders/mnist/test_folder/ --eval_dirs peers/logs/5/RAE-SN_1/GMM_10_sampled/
#python precision-recall-distributions/prd_from_image_folders.py --inception_path /tmp/classify_image_graph_def.pb --reference_dir data_folders/mnist/test_folder/ --eval_dirs peers/logs/6/RAE_1/GMM_10_sampled/
#python precision-recall-distributions/prd_from_image_folders.py --inception_path /tmp/classify_image_graph_def.pb --reference_dir data_folders/mnist/test_folder/ --eval_dirs generated_data/vamp/mnist/
#python precision-recall-distributions/prd_from_image_folders.py --inception_path /tmp/classify_image_graph_def.pb --reference_dir data_folders/mnist/test_folder/ --eval_dirs generated_data/vae/mnist/manifold_sampling/
#python precision-recall-distributions/prd_from_image_folders.py --inception_path /tmp/classify_image_graph_def.pb --reference_dir data_folders/mnist/test_folder/ --eval_dirs generated_data/vae/mnist/gmm/
#python precision-recall-distributions/prd_from_image_folders.py --inception_path /tmp/classify_image_graph_def.pb --reference_dir data_folders/mnist/test_folder/ --eval_dirs generated_data/vae/mnist/gaussian_prior/
#python precision-recall-distributions/prd_from_image_folders.py --inception_path /tmp/classify_image_graph_def.pb --reference_dir data_folders/mnist/test_folder/ --eval_dirs generated_data/ae/mnist/gmm/
#python precision-recall-distributions/prd_from_image_folders.py --inception_path /tmp/classify_image_graph_def.pb --reference_dir data_folders/mnist/test_folder/ --eval_dirs generated_data/ae/mnist/gaussian_prior/
#python precision-recall-distributions/prd_from_image_folders.py --inception_path /tmp/classify_image_graph_def.pb --reference_dir data_folders/mnist/test_folder/ --eval_dirs generated_data/rhvae/mnist/gaussian_prior/
#


################## CIFAR ##################

######## training ######## 

#python peers/Regularized_autoencoders-RAE-/train_raes_vaes.py 8
#python peers/Regularized_autoencoders-RAE-/train_raes_vaes.py 11
#python peers/Regularized_autoencoders-RAE-/train_raes_vaes.py 12
#python peers/Regularized_autoencoders-RAE-/train_raes_vaes.py 13
#python peers/Regularized_autoencoders-RAE-/train_raes_vaes.py 14
#python train_vae.py data_folders/cifar/cifar_10.npz --model 'vamp'
#python train_vae.py data_folders/cifar/cifar_10.npz --model 'vae'
#python train_vae.py data_folders/cifar/cifar_10.npz --model 'ae'
#python train_vae.py data_folders/cifar/cifar_10.npz --model 'rhvae'


######## generation ########

#python peers/Regularized_autoencoders-RAE-/interpolation_fid_and_viz.py 8
#python peers/Regularized_autoencoders-RAE-/interpolation_fid_and_viz.py 11
#python peers/Regularized_autoencoders-RAE-/interpolation_fid_and_viz.py 12
#python peers/Regularized_autoencoders-RAE-/interpolation_fid_and_viz.py 13
#python peers/Regularized_autoencoders-RAE-/interpolation_fid_and_viz.py 14
#python generate_data.py --model_path trained_vae_models/vamp/cifar/best_model.pt --data_path data_folders/cifar/cifar_10.npz --n_samples 10000
#python generate_data.py --model_path trained_vae_models/vae/cifar/best_model.pt --data_path data_folders/cifar/cifar_10.npz --n_samples 10000
#python generate_data.py --model_path trained_vae_models/vae/cifar/best_model.pt --data_path data_folders/cifar/cifar_10.npz --n_samples 10000 --generation 'gauss'
#python generate_data.py --model_path trained_vae_models/vae/cifar/best_model.pt --data_path data_folders/cifar/cifar_10.npz --n_samples 10000 --generation 'gmm'
#python generate_data.py --model_path trained_vae_models/ae/cifar/best_model.pt --data_path data_folders/cifar/cifar_10.npz --n_samples 10000 --generation 'gauss'
#python generate_data.py --model_path trained_vae_models/ae/cifar/best_model.pt --data_path data_folders/cifar/cifar_10.npz --n_samples 10000 --generation 'gmm'
#python generate_data.py --model_path trained_vae_models/rhvae/cifar/best_model.pt --data_path data_folders/cifar/cifar_10.npz --n_samples 10000 --generation 'gauss'


######## metric computation ########

#python TTUR/fid.py peers/logs/8/WAE_1/one_gaussian_sampled/ data_folders/cifar/test_folder --gpu '0'
#python TTUR/fid.py peers/logs/11/RAE-GP_1/GMM_10_sampled/ data_folders/cifar/test_folder --gpu '0'
#python TTUR/fid.py peers/logs/12/RAE-L2_1/GMM_10_sampled/ data_folders/cifar/test_folder --gpu '0'
#python TTUR/fid.py peers/logs/13/RAE-SN_1/GMM_10_sampled/ data_folders/cifar/test_folder --gpu '0'
#python TTUR/fid.py peers/logs/14/RAE_1/GMM_10_sampled/ data_folders/cifar/test_folder --gpu '0'
#python TTUR/fid.py generated_data/vamp/cifar/ data_folders/cifar/test_folder/ --gpu '0'
#python TTUR/fid.py generated_data/vae/cifar/manifold_sampling/ data_folders/cifar/test_folder/ --gpu '0'
#python TTUR/fid.py generated_data/vae/cifar/gmm/ data_folders/cifar/test_folder/ --gpu '0'
#python TTUR/fid.py generated_data/vae/cifar/gaussian_prior/ data_folders/cifar/test_folder/ --gpu '0'
#python TTUR/fid.py generated_data/ae/cifar/gmm/ data_folders/cifar/test_folder/ --gpu '0'
#python TTUR/fid.py generated_data/ae/cifar/gaussian_prior/ data_folders/cifar/test_folder/ --gpu '0'
#python TTUR/fid.py generated_data/rhvae/cifar/gaussian_prior/ data_folders/cifar/test_folder/ --gpu '0'


#python precision-recall-distributions/prd_from_image_folders.py --inception_path /tmp/classify_image_graph_def.pb --reference_dir data_folders/cifar/test_folder/ --eval_dirs peers/logs/8/WAE_1/one_gaussian_sampled/
#python precision-recall-distributions/prd_from_image_folders.py --inception_path /tmp/classify_image_graph_def.pb --reference_dir data_folders/cifar/test_folder/ --eval_dirs peers/logs/11/RAE-GP_1/GMM_10_sampled/
#python precision-recall-distributions/prd_from_image_folders.py --inception_path /tmp/classify_image_graph_def.pb --reference_dir data_folders/cifar/test_folder/ --eval_dirs peers/logs/12/RAE-L2_1/GMM_10_sampled/
#python precision-recall-distributions/prd_from_image_folders.py --inception_path /tmp/classify_image_graph_def.pb --reference_dir data_folders/cifar/test_folder/ --eval_dirs peers/logs/13/RAE-SN_1/GMM_10_sampled/
#python precision-recall-distributions/prd_from_image_folders.py --inception_path /tmp/classify_image_graph_def.pb --reference_dir data_folders/cifar/test_folder/ --eval_dirs peers/logs/14/RAE_1/GMM_10_sampled/
#python precision-recall-distributions/prd_from_image_folders.py --inception_path /tmp/classify_image_graph_def.pb --reference_dir data_folders/cifar/test_folder/ --eval_dirs generated_data/vamp/cifar/
#python precision-recall-distributions/prd_from_image_folders.py --inception_path /tmp/classify_image_graph_def.pb --reference_dir data_folders/cifar/test_folder/ --eval_dirs generated_data/vae/cifar/manifold_sampling/
#python precision-recall-distributions/prd_from_image_folders.py --inception_path /tmp/classify_image_graph_def.pb --reference_dir data_folders/cifar/test_folder/ --eval_dirs generated_data/vae/cifar/gmm/
#python precision-recall-distributions/prd_from_image_folders.py --inception_path /tmp/classify_image_graph_def.pb --reference_dir data_folders/cifar/test_folder/ --eval_dirs generated_data/vae/cifar/gaussian_prior/
#python precision-recall-distributions/prd_from_image_folders.py --inception_path /tmp/classify_image_graph_def.pb --reference_dir data_folders/cifar/test_folder/ --eval_dirs generated_data/ae/cifar/gmm/
#python precision-recall-distributions/prd_from_image_folders.py --inception_path /tmp/classify_image_graph_def.pb --reference_dir data_folders/cifar/test_folder/ --eval_dirs generated_data/ae/cifar/gaussian_prior/
#python precision-recall-distributions/prd_from_image_folders.py --inception_path /tmp/classify_image_graph_def.pb --reference_dir data_folders/cifar/test_folder/ --eval_dirs generated_data/rhvae/cifar/gaussian_prior/



################## CELEBA ##################

######## training ######## 

#python peers/Regularized_autoencoders-RAE-/train_raes_vaes.py 16
#python peers/Regularized_autoencoders-RAE-/train_raes_vaes.py 19
#python peers/Regularized_autoencoders-RAE-/train_raes_vaes.py 20
#python peers/Regularized_autoencoders-RAE-/train_raes_vaes.py 21
#python peers/Regularized_autoencoders-RAE-/train_raes_vaes.py 22
#python train_vae.py data_folders/celeba/ --model 'vamp'
#python train_vae.py data_folders/celeba/ --model 'vae'
#python train_vae.py data_folders/celeba/ --model 'ae'
#python train_vae.py data_folders/celeba/ --model 'rhvae'


######## generation ########

#python peers/Regularized_autoencoders-RAE-/interpolation_fid_and_viz.py 16
#python peers/Regularized_autoencoders-RAE-/interpolation_fid_and_viz.py 19
#python peers/Regularized_autoencoders-RAE-/interpolation_fid_and_viz.py 20
#python peers/Regularized_autoencoders-RAE-/interpolation_fid_and_viz.py 21
#python peers/Regularized_autoencoders-RAE-/interpolation_fid_and_viz.py 22
#python generate_data.py --model_path trained_vae_models/vamp/celeba/best_model.pt --data_path data_folders/celeba/. --n_samples 10000
#python generate_data.py --model_path trained_vae_models/vae/celeba/best_model.pt --data_path data_folders/celeba/. --n_samples 10000
#python generate_data.py --model_path trained_vae_models/vae/celeba/best_model.pt --data_path data_folders/celeba/. --n_samples 10000 --generation 'gauss'
#python generate_data.py --model_path trained_vae_models/vae/celeba/best_model.pt --data_path data_folders/celeba/. --n_samples 10000 --generation 'gmm'
#python generate_data.py --model_path trained_vae_models/ae/celeba/best_model.pt --data_path data_folders/celeba/. --n_samples 10000 --generation 'gauss'
#python generate_data.py --model_path trained_vae_models/ae/celeba/best_model.pt --data_path data_folders/celeba/. --n_samples 10000 --generation 'gmm'
#python generate_data.py --model_path trained_vae_models/rhvae/celeba/best_model.pt --data_path data_folders/celeba/ --n_samples 10000 --generation 'gauss'


######## metric computation ########

#python TTUR/fid.py peers/logs/16/WAE_1/one_gaussian_sampled/ data_folders/celeba/test/test/ --gpu '0'
#python TTUR/fid.py peers/logs/19/RAE-GP_1/GMM_10_sampled/ data_folders/celeba/test/test/ --gpu '0'
#python TTUR/fid.py peers/logs/20/RAE-L2_1/GMM_10_sampled/ data_folders/celeba/test/test/ --gpu '0'
#python TTUR/fid.py peers/logs/21/RAE-SN_1/GMM_10_sampled/ data_folders/celeba/test/test/ --gpu '0'
#python TTUR/fid.py peers/logs/22/RAE_1/GMM_10_sampled/ data_folders/celeba/test/test/ --gpu '0'
#python TTUR/fid.py generated_data/vamp/celeba/ data_folders/celeba/test/test/ --gpu '0'
#python TTUR/fid.py generated_data/vae/celeba/manifold_sampling/ data_folders/celeba/test/test/ --gpu '0'
#python TTUR/fid.py generated_data/vae/celeba/gmm/ data_folders/celeba/test/test/ --gpu '0'
#python TTUR/fid.py generated_data/vae/celeba/gaussian_prior/ data_folders/celeba/test/test/ --gpu '0'
#python TTUR/fid.py generated_data/ae/celeba/gmm/ data_folders/celeba/test/test/ --gpu '0'
#python TTUR/fid.py generated_data/ae/celeba/gaussian_prior/ data_folders/celeba/test/test/ --gpu '0'
#python TTUR/fid.py generated_data/rhvae/celeba/gaussian_prior/ data_folders/celeba/test/test/ --gpu '0'


#python precision-recall-distributions/prd_from_image_folders.py --inception_path /tmp/classify_image_graph_def.pb --reference_dir data_folders/celeba/test/test/ --eval_dirs peers/logs/16/WAE_1/one_gaussian_sampled
#python precision-recall-distributions/prd_from_image_folders.py --inception_path /tmp/classify_image_graph_def.pb --reference_dir data_folders/celeba/test/test/ --eval_dirs peers/logs/19/RAE-GP_1/GMM_10_sampled/
#python precision-recall-distributions/prd_from_image_folders.py --inception_path /tmp/classify_image_graph_def.pb --reference_dir data_folders/celeba/test/test/ --eval_dirs peers/logs/20/RAE-L2_1/GMM_10_sampled/
#python precision-recall-distributions/prd_from_image_folders.py --inception_path /tmp/classify_image_graph_def.pb --reference_dir data_folders/celeba/test/test/ --eval_dirs peers/logs/21/RAE-SN_1/GMM_10_sampled/
#python precision-recall-distributions/prd_from_image_folders.py --inception_path /tmp/classify_image_graph_def.pb --reference_dir data_folders/celeba/test/test/ --eval_dirs peers/logs/22/RAE_1/GMM_10_sampled/
#python precision-recall-distributions/prd_from_image_folders.py --inception_path /tmp/classify_image_graph_def.pb --reference_dir data_folders/celeba/test/test/ --eval_dirs generated_data/vamp/celeba/
#python precision-recall-distributions/prd_from_image_folders.py --inception_path /tmp/classify_image_graph_def.pb --reference_dir data_folders/celeba/test/test/ --eval_dirs generated_data/vae/celeba/manifold_sampling/
#python precision-recall-distributions/prd_from_image_folders.py --inception_path /tmp/classify_image_graph_def.pb --reference_dir data_folders/celeba/test/test/ --eval_dirs generated_data/vae/celeba/gmm/
#python precision-recall-distributions/prd_from_image_folders.py --inception_path /tmp/classify_image_graph_def.pb --reference_dir data_folders/celeba/test/test/ --eval_dirs generated_data/vae/celeba/gaussian_prior/
#python precision-recall-distributions/prd_from_image_folders.py --inception_path /tmp/classify_image_graph_def.pb --reference_dir data_folders/celeba/test/test/ --eval_dirs generated_data/ae/celeba/gmm/
#python precision-recall-distributions/prd_from_image_folders.py --inception_path /tmp/classify_image_graph_def.pb --reference_dir data_folders/celeba/test/test/ --eval_dirs generated_data/ae/celeba/gaussian_prior/
#python precision-recall-distributions/prd_from_image_folders.py --inception_path /tmp/classify_image_graph_def.pb --reference_dir data_folders/celeba/test/test/ --eval_dirs generated_data/rhvae/celeba/gaussian_prior/



################## SVHN ##################

######## training ######## 

#python peers/Regularized_autoencoders-RAE-/train_raes_vaes.py 24
#python peers/Regularized_autoencoders-RAE-/train_raes_vaes.py 27
#python peers/Regularized_autoencoders-RAE-/train_raes_vaes.py 28
#python peers/Regularized_autoencoders-RAE-/train_raes_vaes.py 29
#python peers/Regularized_autoencoders-RAE-/train_raes_vaes.py 30
#python train_vae.py data_folders/svhn/train_32x32.mat --model 'vamp'
#python train_vae.py data_folders/svhn/train_32x32.mat --model 'vae'
#python train_vae.py data_folders/svhn/train_32x32.mat --model 'ae'
#python train_vae.py data_folders/svhn/train_32x32.mat --model 'rhvae'


######## generation ########

#python peers/Regularized_autoencoders-RAE-/interpolation_fid_and_viz.py 24
#python peers/Regularized_autoencoders-RAE-/interpolation_fid_and_viz.py 27
#python peers/Regularized_autoencoders-RAE-/interpolation_fid_and_viz.py 28
#python peers/Regularized_autoencoders-RAE-/interpolation_fid_and_viz.py 29
#python peers/Regularized_autoencoders-RAE-/interpolation_fid_and_viz.py 30
#python generate_data.py --model_path trained_vae_models/vamp/svhn/best_model.pt --data_path data_folders/svhn/train_32x32.mat --n_samples 10000
#python generate_data.py --model_path trained_vae_models/vae/svhn/best_model.pt --data_path data_folders/svhn/train_32x32.mat --n_samples 10000
#python generate_data.py --model_path trained_vae_models/vae/svhn/best_model.pt --data_path data_folders/svhn/train_32x32.mat --n_samples 10000 --generation 'gauss'
#python generate_data.py --model_path trained_vae_models/vae/svhn/best_model.pt --data_path data_folders/svhn/train_32x32.mat --n_samples 10000 --generation 'gmm'
#python generate_data.py --model_path trained_vae_models/ae/svhn/best_model.pt --data_path data_folders/svhn/train_32x32.mat --n_samples 10000 --generation 'gauss'
#python generate_data.py --model_path trained_vae_models/ae/svhn/best_model.pt --data_path data_folders/svhn/train_32x32.mat --n_samples 10000 --generation 'gmm'
#python generate_data.py --model_path trained_vae_models/rhvae/svhn/best_model.pt --data_path data_folders/svhn/train_32x32.mat --n_samples 10000 --generation 'gauss'


######## metric computation ########

#python TTUR/fid.py peers/logs/24/WAE_1/one_gaussian_sampled/ data_folders/svhn/test_folder/ --gpu '0'
#python TTUR/fid.py peers/logs/27/RAE-GP_1/GMM_10_sampled/ data_folders/svhn/test_folder/ --gpu '0'
#python TTUR/fid.py peers/logs/28/RAE-L2_1/GMM_10_sampled/ data_folders/svhn/test_folder/ --gpu '0'
#python TTUR/fid.py peers/logs/29/RAE-SN_1/GMM_10_sampled/ data_folders/svhn/test_folder/ --gpu '0'
#python TTUR/fid.py peers/logs/30/RAE_1/GMM_10_sampled/ data_folders/svhn/test_folder/ --gpu '0'
#python TTUR/fid.py generated_data/vamp/svhn/ data_folders/svhn/test_folder/ --gpu '0'
#python TTUR/fid.py generated_data/vae/svhn/manifold_sampling/ data_folders/svhn/test_folder/ --gpu '0'
#python TTUR/fid.py generated_data/vae/svhn/gmm/ data_folders/svhn/test_folder/ --gpu '0'
#python TTUR/fid.py generated_data/vae/svhn/gaussian_prior/ data_folders/svhn/test_folder/ --gpu '0'
#python TTUR/fid.py generated_data/ae/svhn/gmm/ data_folders/svhn/test_folder/ --gpu '0'
#python TTUR/fid.py generated_data/ae/svhn/gaussian_prior/ data_folders/svhn/test_folder/ --gpu '0'
#python TTUR/fid.py generated_data/rhvae/svhn/gaussian_prior/ data_folders/svhn/test_folder/ --gpu '0'

#python precision-recall-distributions/prd_from_image_folders.py --inception_path /tmp/classify_image_graph_def.pb --reference_dir data_folders/svhn/test_folder --eval_dirs peers/logs/24/WAE_1/one_gaussian_sampled
#python precision-recall-distributions/prd_from_image_folders.py --inception_path /tmp/classify_image_graph_def.pb --reference_dir data_folders/svhn/test_folder --eval_dirs peers/logs/27/RAE-GP_1/GMM_10_sampled/
#python precision-recall-distributions/prd_from_image_folders.py --inception_path /tmp/classify_image_graph_def.pb --reference_dir data_folders/svhn/test_folder --eval_dirs peers/logs/28/RAE-L2_1/GMM_10_sampled/
#python precision-recall-distributions/prd_from_image_folders.py --inception_path /tmp/classify_image_graph_def.pb --reference_dir data_folders/svhn/test_folder --eval_dirs peers/logs/29/RAE-SN_1/GMM_10_sampled/
#python precision-recall-distributions/prd_from_image_folders.py --inception_path /tmp/classify_image_graph_def.pb --reference_dir data_folders/svhn/test_folder --eval_dirs peers/logs/30/RAE_1/GMM_10_sampled/
#python precision-recall-distributions/prd_from_image_folders.py --inception_path /tmp/classify_image_graph_def.pb --reference_dir data_folders/svhn/test_folder --eval_dirs generated_data/vamp/svhn/
#python precision-recall-distributions/prd_from_image_folders.py --inception_path /tmp/classify_image_graph_def.pb --reference_dir data_folders/svhn/test_folder --eval_dirs generated_data/vae/svhn/manifold_sampling/
#python precision-recall-distributions/prd_from_image_folders.py --inception_path /tmp/classify_image_graph_def.pb --reference_dir data_folders/svhn/test_folder --eval_dirs generated_data/vae/svhn/gmm/
#python precision-recall-distributions/prd_from_image_folders.py --inception_path /tmp/classify_image_graph_def.pb --reference_dir data_folders/svhn/test_folder --eval_dirs generated_data/vae/svhn/gaussian_prior/
#python precision-recall-distributions/prd_from_image_folders.py --inception_path /tmp/classify_image_graph_def.pb --reference_dir data_folders/svhn/test_folder --eval_dirs generated_data/ae/svhn/gmm/
#python precision-recall-distributions/prd_from_image_folders.py --inception_path /tmp/classify_image_graph_def.pb --reference_dir data_folders/svhn/test_folder --eval_dirs generated_data/ae/svhn/gaussian_prior/
#python precision-recall-distributions/prd_from_image_folders.py --inception_path /tmp/classify_image_graph_def.pb --reference_dir data_folders/svhn/test_folder --eval_dirs generated_data/rhvae/svhn/gaussian_prior/
