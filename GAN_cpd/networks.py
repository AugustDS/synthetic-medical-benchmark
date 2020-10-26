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

import numpy as np
import tensorflow as tf

#----------------------------------------------------------------------------
# Get/create weight tensor for a convolution or fully-connected layer.

def get_weight(shape, gain=1, use_wscale=True, lrmul=1, weight_var='weight'):
    fan_in = np.prod(shape[:-1]) # [kernel, kernel, fmaps_in, fmaps_out] or [in, out]
    he_std = gain / np.sqrt(fan_in) # He init

    # Equalized learning rate and custom learning rate multiplier.
    if use_wscale:
        init_std = 1.0 / lrmul
        runtime_coef = he_std * lrmul
    else:
        init_std = he_std / lrmul
        runtime_coef = lrmul

    # Create variable.
    init = tf.initializers.random_normal(0, init_std)
    return tf.get_variable(weight_var, shape=shape, initializer=init) * runtime_coef

#----------------------------------------------------------------------------
# Fully-connected layer.

def dense_layer(x, fmaps, gain=1, use_wscale=True, lrmul=1, weight_var='weight'):
    if len(x.shape) > 2:
        x = tf.reshape(x, [-1, np.prod([d.value for d in x.shape[1:]])])
    w = get_weight([x.shape[1].value, fmaps], gain=gain, use_wscale=use_wscale, lrmul=lrmul, weight_var=weight_var)
    w = tf.cast(w, x.dtype)
    return tf.matmul(x, w)

#----------------------------------------------------------------------------
# Fused upscale2d + conv2d.
# Faster and uses less memory than performing the operations separately.

def upscale2d_conv2d(x, fmaps, kernel, gain=np.sqrt(2), use_wscale=False):
    assert kernel >= 1 and kernel % 2 == 1
    w = get_weight([kernel, kernel, fmaps, x.shape[1].value], gain=gain, use_wscale=use_wscale)
    w = tf.pad(w, [[1,1], [1,1], [0,0], [0,0]], mode='CONSTANT')
    w = tf.add_n([w[1:, 1:], w[:-1, 1:], w[1:, :-1], w[:-1, :-1]])
    w = tf.cast(w, x.dtype)
    os = [tf.shape(x)[0], fmaps, x.shape[2] * 2, x.shape[3] * 2]
    return tf.nn.conv2d_transpose(x, w, os, strides=[1,1,2,2], padding='SAME', data_format='NCHW')

#----------------------------------------------------------------------------
# Fused conv2d + downscale2d.
# Faster and uses less memory than performing the operations separately.

def conv2d_downscale2d(x, fmaps, kernel, gain=np.sqrt(2), use_wscale=False):
    assert kernel >= 1 and kernel % 2 == 1
    w = get_weight([kernel, kernel, x.shape[1].value, fmaps], gain=gain, use_wscale=use_wscale)
    w = tf.pad(w, [[1,1], [1,1], [0,0], [0,0]], mode='CONSTANT')
    w = tf.add_n([w[1:, 1:], w[:-1, 1:], w[1:, :-1], w[:-1, :-1]]) * 0.25
    w = tf.cast(w, x.dtype)
    return tf.nn.conv2d(x, w, strides=[1,1,2,2], padding='SAME', data_format='NCHW')


#----------------------------------------------------------------------------
# Convolution layer with optional upsampling or downsampling.

def conv2d_layer(x, fmaps, kernel, up=False, down=False, gain=1, use_wscale=True, lrmul=1, weight_var='weight'):
    assert not (up and down)
    assert kernel >= 1 and kernel % 2 == 1
    if up:
        x = upscale2d_conv2d(x, fmaps, kernel=kernel, use_wscale=use_wscale)
    elif down:
        x = conv2d_downscale2d(x, fmaps, kernel=kernel, use_wscale=use_wscale)
    else:
        w = get_weight([kernel, kernel, x.shape[1].value, fmaps], gain=gain, use_wscale=use_wscale, lrmul=lrmul, weight_var=weight_var)
        x = tf.nn.conv2d(x, tf.cast(w, x.dtype), data_format='NCHW', strides=[1,1,1,1], padding='SAME')
    return x

#----------------------------------------------------------------------------
# Box filter downscaling layer.

def downscale2d(x, factor=2):
    assert isinstance(factor, int) and factor >= 1
    if factor == 1: return x
    with tf.variable_scope('Downscale2D'):
        ksize = [1, 1, factor, factor]
        return tf.nn.avg_pool(x, ksize=ksize, strides=ksize, padding='VALID', data_format='NCHW') # NOTE: requires tf_config['graph_options.place_pruned_graph'] = True

def upscale2d(x, factor=2):
    assert isinstance(factor, int) and factor >= 1
    if factor == 1: return x
    with tf.variable_scope('Upscale2D'):
        s = x.shape
        x = tf.reshape(x, [-1, s[1], s[2], 1, s[3], 1])
        x = tf.tile(x, [1, 1, 1, factor, 1, factor])
        x = tf.reshape(x, [-1, s[1], s[2] * factor, s[3] * factor])
        return x

def apply_bias(x):
    b = tf.get_variable('bias', shape=[x.shape[1]], initializer=tf.initializers.zeros())
    b = tf.cast(b, x.dtype)
    if len(x.shape) == 2:
        return x + b
    else:
        return x + tf.reshape(b, [1, -1, 1, 1])

#----------------------------------------------------------------------------
# Leaky ReLU activation. Same as tf.nn.leaky_relu, but supports FP16.

def leaky_relu(x, alpha=0.2):
    with tf.name_scope('LeakyRelu'):
        alpha = tf.constant(alpha, dtype=x.dtype, name='alpha')
        return tf.maximum(x * alpha, x)

#----------------------------------------------------------------------------
# Minibatch standard deviation layer.

def minibatch_stddev_layer(x, group_size=4, num_new_features=1):
    group_size = tf.minimum(group_size, tf.shape(x)[0])     # Minibatch must be divisible by (or smaller than) group_size.
    s = x.shape                                             # [NCHW]  Input shape.
    y = tf.reshape(x, [group_size, -1, num_new_features, s[1]//num_new_features, s[2], s[3]])   # [GMncHW] Split minibatch into M groups of size G. Split channels into n channel groups c.
    y = tf.cast(y, tf.float32)                              # [GMncHW] Cast to FP32.
    y -= tf.reduce_mean(y, axis=0, keepdims=True)           # [GMncHW] Subtract mean over group.
    y = tf.reduce_mean(tf.square(y), axis=0)                # [MncHW]  Calc variance over group.
    y = tf.sqrt(y + 1e-8)                                   # [MncHW]  Calc stddev over group.
    y = tf.reduce_mean(y, axis=[2,3,4], keepdims=True)      # [Mn111]  Take average over fmaps and pixels.
    y = tf.reduce_mean(y, axis=[2])                         # [Mn11] Split channels into c channel groups
    y = tf.cast(y, x.dtype)                                 # [Mn11]  Cast back to original data type.
    y = tf.tile(y, [group_size, 1, s[2], s[3]])             # [NnHW]  Replicate over group and pixels.
    return tf.concat([x, y], axis=1)                        # [NCHW]  Append as new fmap.

def pixel_norm(x, epsilon=1e-8):
    with tf.variable_scope('PixelNorm'):
        return x * tf.rsqrt(tf.reduce_mean(tf.square(x), axis=1, keepdims=True) + epsilon)

def conditional_pixel_norm(x,z,epsilon=1e-8):
    with tf.variable_scope('PixelNorm'):
        _, c, _, _ = x.get_shape().as_list()

        with tf.variable_scope("beta"):
            beta = apply_bias(dense_layer(z, fmaps=c, use_wscale=True))  #[BS,c]
        with tf.variable_scope("gamma"):
            gamma = apply_bias(dense_layer(z, fmaps=c, use_wscale=True)) #[BS,c]

        beta  = tf.reshape(beta,[-1,c,1,1])
        gamma = tf.reshape(gamma,[-1,c,1,1])
        return x * tf.rsqrt(tf.reduce_mean(tf.square(x), axis=1, keepdims=True) + epsilon)*gamma + beta 

def G_paper(
    latents_in,                         # First input: Latent vectors (Z) [minibatch, latent_size].
    labels_in,                          # Second input: Conditioning labels [minibatch, label_size].
    latent_size         = None,         # Dimensionality of the latent vectors. None = min(fmap_base, fmap_max).
    label_size          = 14,           # Dimensionality of the labels, 0 if no labels. Overridden based on dataset.
    num_channels        = 1,            # Number of output color channels.
    resolution          = 1024,         # Output resolution.
    fmap_base           = 16 << 10,     # Overall multiplier for the number of feature maps.
    fmap_decay          = 1.0,          # log2 feature map reduction when doubling the resolution.
    fmap_min            = 1,            # Minimum number of feature maps in any layer.
    fmap_max            = 512,          # Maximum number of feature maps in any layer.
    architecture        = 'skip',       # Architecture: 'orig', 'skip', 'resnet'.
    dtype               = 'float32',    # Data type to use for activations and outputs.
    use_pixelnorm       = True,         # Enable pixelwise feature vector normalization?
    normalize_latents   = True,
    pixelnorm_epsilon   = 1e-8,         # Constant epsilon for pixelwise feature vector normalization.
    **_kwargs):                         # Ignore unrecognized keyword args.

    resolution_log2 = int(np.log2(resolution))
    assert resolution == 2**resolution_log2 and resolution >= 4
    def nf(stage): return np.clip(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_min, fmap_max)
    def PN(x): return conditional_pixel_norm(x, combo_in, epsilon=pixelnorm_epsilon) if use_pixelnorm else x
    assert architecture in ['orig', 'skip', 'resnet']
    act = leaky_relu 
    num_layers = resolution_log2 * 2 - 2
    images_out = None

    # Primary inputs.
    if latent_size is None: latent_size = nf(0)
    latents_in.set_shape([None, latent_size])
    latents_in = tf.cast(latents_in, dtype)
    labels_in.set_shape([None, label_size])
    labels_in = tf.cast(labels_in, dtype)
    combo_in = tf.cast(tf.concat([latents_in, labels_in], axis=1), dtype)

    # Building blocks for main layers.
    def block(x, res): # res = 3..resolution_log2
        t = x
        with tf.variable_scope('Conv0_up'):
            x = PN(act(apply_bias(conv2d_layer(x, fmaps=nf(res-1), up=True, kernel=3))))
        with tf.variable_scope('Conv1'):
            x = PN(act(apply_bias(conv2d_layer(x, fmaps=nf(res-1), kernel=3))))
        if architecture == 'resnet':
            with tf.variable_scope('Skip'):
                t = conv2d_layer(t, fmaps=nf(res-1), kernel=1, up=True)
                x = (x + t) * (1 / np.sqrt(2))
        return x

    def torgb(x, y, res): # res = 2..resolution_log2
        with tf.variable_scope('ToRGB'):
            t = apply_bias(conv2d_layer(x, fmaps=num_channels, kernel=1))
            return t if y is None else y + t

    # Early layers.
    y = None
    x = combo_in
    if normalize_latents: 
        x = pixel_norm(x, epsilon=pixelnorm_epsilon)
    with tf.variable_scope('4x4'):
        with tf.variable_scope('Dense'):
            x = dense_layer(x, fmaps=nf(1)*16)
            x = tf.reshape(x, [-1, nf(1), 4, 4])
            x = PN(act(apply_bias(x)))
        with tf.variable_scope('Conv'):
            x = PN(act(apply_bias(conv2d_layer(x, fmaps=nf(1), kernel=3))))
        if architecture == 'skip':
            y = torgb(x, y, 2)

    # Main layers.
    for res in range(3, resolution_log2 + 1):
        with tf.variable_scope('%dx%d' % (2**res, 2**res)):
            x = block(x, res)
            if architecture == 'skip':
                y = upscale2d(y)
            if architecture == 'skip' or res == resolution_log2:
                y = torgb(x, y, res)
    images_out = y

    assert images_out.dtype == tf.as_dtype(dtype)
    return tf.identity(images_out, name='images_out')


def D_paper(
    images_in,                          # First input: Images [minibatch, channel, height, width].
    labels_in,                          # Second input: Conditioning labels [minibatch, label_size].
    num_channels        = 1,            # Number of input color channels. Overridden based on dataset.
    resolution          = 32,           # Input resolution. Overridden based on dataset.
    label_size          = 0,            # Dimensionality of the labels, 0 if no labels. Overridden based on dataset.
    fmap_base           = 16 << 10,     # Overall multiplier for the number of feature maps.
    fmap_decay          = 1.0,          # log2 feature map reduction when doubling the resolution.
    fmap_min            = 1,            # Minimum number of feature maps in any layer.
    fmap_max            = 512,          # Maximum number of feature maps in any layer.
    architecture        = 'resnet',     # Architecture: 'orig', 'skip', 'resnet'.
    mbstd_group_size    = 4,            # Group size for the minibatch standard deviation layer, 0 = disable.
    mbstd_num_features  = 1,            # Number of features for the minibatch standard deviation layer.
    dtype               = 'float32',    # Data type to use for activations and outputs.
    **_kwargs):                         # Ignore unrecognized keyword args.

    resolution_log2 = int(np.log2(resolution))
    assert resolution == 2**resolution_log2 and resolution >= 4
    def nf(stage): return np.clip(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_min, fmap_max)
    assert architecture in ['orig', 'skip', 'resnet']
    act = leaky_relu

    labels_in.set_shape([None, label_size])
    labels_in = tf.cast(labels_in, dtype)

    images_in.set_shape([None, num_channels, resolution, resolution])
    images_in = tf.cast(images_in, dtype)

    # Building blocks for main layers.
    def fromrgb(x, y, res): # res = 2..resolution_log2
        with tf.variable_scope('FromRGB'):
            t = act(apply_bias(conv2d_layer(y, fmaps=nf(res-1), kernel=1)))
            return t if x is None else x + t
    def block(x, res): # res = 2..resolution_log2
        t = x
        with tf.variable_scope('Conv0'):
            x = act(apply_bias(conv2d_layer(x, fmaps=nf(res-1), kernel=3)))
        with tf.variable_scope('Conv1_down'):
            x = act(apply_bias(conv2d_layer(x, fmaps=nf(res-2), kernel=3, down=True)))
        if architecture == 'resnet':
            with tf.variable_scope('Skip'):
                t = conv2d_layer(t, fmaps=nf(res-2), kernel=1, down=True)
                x = (x + t) * (1 / np.sqrt(2))
        return x

    # Main layers.
    x = None
    y = images_in
    for res in range(resolution_log2, 2, -1):
        with tf.variable_scope('%dx%d' % (2**res, 2**res)):
            if architecture == 'skip' or res == resolution_log2:
                x = fromrgb(x, y, res)
            x = block(x, res)
            if architecture == 'skip':
                y = downscale2d(y)

    # Final layers.
    with tf.variable_scope('4x4'):
        if architecture == 'skip':
            x = fromrgb(x, y, 2)
        if mbstd_group_size > 1:
            with tf.variable_scope('MinibatchStddev'):
                x = minibatch_stddev_layer(x, mbstd_group_size, mbstd_num_features)
        with tf.variable_scope('Conv'):
            x = act(apply_bias(conv2d_layer(x, fmaps=nf(1), kernel=3)))
        with tf.variable_scope('Dense0'):
            x = act(apply_bias(dense_layer(x, fmaps=nf(0))))

        # Output layer with label conditioning from "Which Training Methods for GANs do actually Converge?"
    with tf.variable_scope('Output'):
        x = apply_bias(dense_layer(x, fmaps=max(labels_in.shape[1], 1)))
        if labels_in.shape[1] > 0:
            x = tf.reduce_sum(x * labels_in, axis=1, keepdims=True)
    scores_out = x

    assert scores_out.dtype == tf.as_dtype(dtype)
    scores_out = tf.identity(scores_out, name='scores_out')
    return scores_out


