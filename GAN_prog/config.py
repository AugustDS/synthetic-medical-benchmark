# Copyright 2018, Tero Karras, NVIDIA CORPORATION
#
# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
#
# Adapted from the original implementation by Tero Karras.
# Source https://github.com/tkarras/progressive_growing_of_gans

#----------------------------------------------------------------------------
# Convenience class that behaves exactly like dict(), but allows accessing
# the keys and values using the attribute syntax, i.e., "mydict.key = value".

class EasyDict(dict):
    def __init__(self, *args, **kwargs): super().__init__(*args, **kwargs)
    def __getattr__(self, name): return self[name]
    def __setattr__(self, name, value): self[name] = value
    def __delattr__(self, name): del self[name]

#----------------------------------------------------------------------------
# Paths.
# These will be replaced by correct paths from parser when calling train

data_dir = "/overwritten"
result_dir = "/overwritten"

#----------------------------------------------------------------------------
# TensorFlow options.

tf_config = EasyDict()  # TensorFlow session config, set by tfutil.init_tf().
env = EasyDict()        # Environment variables, set by the main program in train.py.

tf_config['graph_options.place_pruned_graph']   = True      # False (default) = Check that all ops are available on the designated device. True = Skip the check for ops that are not used.
#tf_config['gpu_options.allow_growth']          = False     # False (default) = Allocate all GPU memory at the beginning. True = Allocate only as much GPU memory as needed.
#env.CUDA_VISIBLE_DEVICES                       = '0'       # Unspecified (default) = Use all available GPUs. List of ints = CUDA device numbers to use.
env.TF_CPP_MIN_LOG_LEVEL                        = '1'       # 0 (default) = Print all available debug info from TensorFlow. 1 = Print warnings and errors, but disable debug info.

#----------------------------------------------------------------------------
# Official training configs, targeted mainly for CelebA-HQ.
# To run, comment/uncomment the lines as appropriate and launch train.py.

desc        = 'pgan'                                        # Description string included in result subdir name.
random_seed = 1000                                          # Global random seed.
dataset     = EasyDict()                                    # Options for dataset.load_dataset().
train       = EasyDict(func='train.train_progressive_gan')  # Options for main training func.
G           = EasyDict(func='networks.G_paper')             # Options for generator network.
D           = EasyDict(func='networks.D_paper')             # Options for discriminator network.
G_opt       = EasyDict(beta1=0.0, beta2=0.99, epsilon=1e-8) # Options for generator optimizer.
D_opt       = EasyDict(beta1=0.0, beta2=0.99, epsilon=1e-8) # Options for discriminator optimizer.
G_loss      = EasyDict(func='loss.G_wgan_acgan')            # Options for generator loss.
D_loss      = EasyDict(func='loss.D_wgangp_acgan')          # Options for discriminator loss.
sched       = EasyDict()                                    # Options for train.TrainingSchedule.
grid        = EasyDict(size='1080p', layout='random')       # Options for train.setup_snapshot_image_grid().

# Dataset (choose one).
desc += '-xray'; dataset = EasyDict(tfrecord_dir='overwritten'); train.mirror_augment = False

# Conditioning & snapshot options.
desc += '-cond'; dataset.max_label_size = 'full'; # conditioned on full label

#----------------------------------------------------------------------------
# Config presets (choose one).
#----------------------------------------------------------------------------
# 32x32 resolution | Default: '-preset-v2-1gpu'
desc += '-preset-v2-1gpu';  num_gpus = 1; sched.minibatch_base = 4;  sched.lod_initial_resolution = 8; sched.minibatch_dict = {8: 256, 16: 128, 32: 64}; train.total_kimg = 12000; sched.lod_training_kimg = 1400; sched.lod_transition_kimg = 1400; train.compute_fid_score = True; train.minimum_fid_kimg = 7000; train.fid_snapshot_ticks = 4; train.fid_patience=2; #train.resume_run_id="/path_to_network_snapshot/network-snapshot-000000.pkl"; train.resume_kimg=000000;
#desc += '-preset-v2-2gpus'; num_gpus = 2; sched.minibatch_base = 8;  sched.lod_initial_resolution = 8; sched.minibatch_dict = {8: 256, 16: 128, 32: 64}; train.total_kimg = 12000; sched.lod_training_kimg = 1400; sched.lod_transition_kimg = 1400; train.compute_fid_score = True; train.minimum_fid_kimg = 7000; train.fid_snapshot_ticks = 4; train.fid_patience=2; #train.resume_run_id="/path_to_network_snapshot/network-snapshot-000000.pkl"; train.resume_kimg=000000;
#desc += '-preset-v2-4gpus'; num_gpus = 4; sched.minibatch_base = 16; sched.lod_initial_resolution = 8; sched.minibatch_dict = {8: 256, 16: 128, 32: 64}; train.total_kimg = 12000; sched.lod_training_kimg = 1400; sched.lod_transition_kimg = 1400; train.compute_fid_score = True; train.minimum_fid_kimg = 7000; train.fid_snapshot_ticks = 4; train.fid_patience=2; #train.resume_run_id="/path_to_network_snapshot/network-snapshot-000000.pkl"; train.resume_kimg=000000;

# 64x64 resolution | Default: '-preset-v2-2gpus'
#desc += '-preset-v2-2gpus'; num_gpus = 2; sched.minibatch_base = 8;  sched.lod_initial_resolution = 8; sched.minibatch_dict = {8: 256, 16: 128, 32: 64, 64:32}; train.total_kimg = 14000; sched.lod_training_kimg = 1000; sched.lod_transition_kimg = 1000; train.compute_fid_score = True; train.minimum_fid_kimg = 7000; train.fid_snapshot_ticks = 4; train.fid_patience=2; #train.resume_run_id="/path_to_network_snapshot/network-snapshot-000000.pkl"; train.resume_kimg=000000; 
#desc += '-preset-v2-4gpus'; num_gpus = 4; sched.minibatch_base = 16; sched.lod_initial_resolution = 8; sched.minibatch_dict = {8: 256, 16: 128, 32: 64, 64:32}; train.total_kimg = 14000; sched.lod_training_kimg = 1000; sched.lod_transition_kimg = 1000; train.compute_fid_score = True; train.minimum_fid_kimg = 7000; train.fid_snapshot_ticks = 4; train.fid_patience=2; #train.resume_run_id="/path_to_network_snapshot/network-snapshot-000000.pkl"; train.resume_kimg=000000; 
#desc += '-preset-v2-8gpus'; num_gpus = 8; sched.minibatch_base = 32; sched.lod_initial_resolution = 8; sched.minibatch_dict = {8: 256, 16: 128, 32: 64, 64:32}; train.total_kimg = 14000; sched.lod_training_kimg = 1000; sched.lod_transition_kimg = 1000; train.compute_fid_score = True; train.minimum_fid_kimg = 7000; train.fid_snapshot_ticks = 4; train.fid_patience=2; #train.resume_run_id="/path_to_network_snapshot/network-snapshot-000000.pkl"; train.resume_kimg=000000; 

# 128x128 resolution | Default: '-preset-v2-2gpus'
#desc += '-preset-v2-2gpus'; num_gpus = 2; sched.minibatch_base = 8;  sched.lod_initial_resolution = 8; sched.minibatch_dict = {8: 256, 16: 128, 32: 64, 64:32, 128:16}; train.total_kimg = 16000; sched.lod_training_kimg = 1000; sched.lod_transition_kimg = 1000; train.compute_fid_score = True; train.minimum_fid_kimg = 9000; train.fid_snapshot_ticks = 4; train.fid_patience=2; #train.resume_run_id="/path_to_network_snapshot/network-snapshot-000000.pkl"; train.resume_kimg=000000; 
#desc += '-preset-v2-4gpus'; num_gpus = 4; sched.minibatch_base = 16; sched.lod_initial_resolution = 8; sched.minibatch_dict = {8: 256, 16: 128, 32: 64, 64:32, 128:16}; train.total_kimg = 16000; sched.lod_training_kimg = 1000; sched.lod_transition_kimg = 1000; train.compute_fid_score = True; train.minimum_fid_kimg = 9000; train.fid_snapshot_ticks = 4; train.fid_patience=2; #train.resume_run_id="/path_to_network_snapshot/network-snapshot-000000.pkl"; train.resume_kimg=000000; 
#desc += '-preset-v2-8gpus'; num_gpus = 8; sched.minibatch_base = 32; sched.lod_initial_resolution = 8; sched.minibatch_dict = {8: 256, 16: 128, 32: 64, 64:32, 128:16}; train.total_kimg = 16000; sched.lod_training_kimg = 1000; sched.lod_transition_kimg = 1000; train.compute_fid_score = True; train.minimum_fid_kimg = 9000; train.fid_snapshot_ticks = 4; train.fid_patience=2; #train.resume_run_id="/path_to_network_snapshot/network-snapshot-000000.pkl"; train.resume_kimg=000000; 
#----------------------------------------------------------------------------

# Numerical precision (choose one).
desc += '-fp32'; sched.max_minibatch_per_gpu = {256: 16}
#desc += '-fp16'; G.dtype = 'float16'; D.dtype = 'float16'; G.pixelnorm_epsilon=1e-4; G_opt.use_loss_scaling = True; D_opt.use_loss_scaling = True; sched.max_minibatch_per_gpu = {512: 16, 1024: 8}

# Special modes.
#desc += '-GRAPH'; train.save_tf_graph = True
#desc += '-HIST'; train.save_weight_histograms = True

#----------------------------------------------------------------------------
# Utility scripts.
# To run, uncomment the appropriate line and launch train.py.

#train = EasyDict(func='util_scripts.generate_fake_images', run_id=23, num_pngs=1000); num_gpus = 1; desc = 'fake-images-' + str(train.run_id)
#train = EasyDict(func='util_scripts.generate_fake_images', run_id=23, grid_size=[15,8], num_pngs=10, image_shrink=4); num_gpus = 1; desc = 'fake-grids-' + str(train.run_id)
#train = EasyDict(func='util_scripts.generate_interpolation_video', run_id=23, grid_size=[1,1], duration_sec=60.0, smoothing_sec=1.0); num_gpus = 1; desc = 'interpolation-video-' + str(train.run_id)
#train = EasyDict(func='util_scripts.generate_training_video', run_id=23, duration_sec=20.0); num_gpus = 1; desc = 'training-video-' + str(train.run_id)

#train = EasyDict(func='util_scripts.evaluate_metrics', run_id=23, log='metric-swd-16k.txt', metrics=['swd'], num_images=16384, real_passes=2); num_gpus = 1; desc = train.log.split('.')[0] + '-' + str(train.run_id)
#train = EasyDict(func='util_scripts.evaluate_metrics', run_id="000", log='metric-fid-10k.txt', metrics=['fid'], num_images=10000, real_passes=1); num_gpus = 1; desc = train.log.split('.')[0] + '-' + str(train.run_id)
#train = EasyDict(func='util_scripts.evaluate_metrics', run_id=23, log='metric-fid-50k.txt', metrics=['fid'], num_images=50000, real_passes=1); num_gpus = 1; desc = train.log.split('.')[0] + '-' + str(train.run_id)
#train = EasyDict(func='util_scripts.evaluate_metrics', run_id=23, log='metric-is-50k.txt', metrics=['is'], num_images=50000, real_passes=1); num_gpus = 1; desc = train.log.split('.')[0] + '-' + str(train.run_id)
#train = EasyDict(func='util_scripts.evaluate_metrics', run_id=23, log='metric-msssim-20k.txt', metrics=['msssim'], num_images=20000, real_passes=1); num_gpus = 1; desc = train.log.split('.')[0] + '-' + str(train.run_id)

#----------------------------------------------------------------------------
