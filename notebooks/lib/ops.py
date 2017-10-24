import tensorflow as tf

def conv2d( name, # layer name
    images, # input tensor
    image_width, image_height, n_channels, # input tensor shape
    kernal_width, kernal_height, feature_depth, # kernal shape
    h_slide, v_slide, padding, # sliding
    biased=True): # wether add bias or not
    
    print("generating convolution layer:{}".format(name))
    
    # weights
    weights = tf.get_variable(
        "weights_{}".format(name),
        [kernal_width, kernal_height, n_channels, feature_depth],
        initializer=tf.uniform_unit_scaling_initializer(seed=GLOBAL_RAND_SEED())
    )

    # convolution
    conv = tf.nn.conv2d(
        images,
        weights,
        strides=[1, h_slide, v_slide, 1],
        padding=padding
    )

    # bias
    bias = None
    if biased:
        bias = tf.get_variable(
            "bias_{}".format(name),
            [feature_depth],
            initializer=tf.constant_initializer(0.0)
        )

        conv = tf.nn.bias_add(conv, bias)

    return conv, weights, bias

# utils & ops
def merge_conf(conf, default_conf):
    result = default_conf.copy()
    for key in conf:
        result[key] = conf[key]
        
    return result

def cross_entropy(x, y):
    return tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)
    )

def conv2d_norm(name, inputs, kernal_size, output_depth, activation=tf.nn.relu, more=False):
    # input shape
    input_shape = inputs.get_shape().as_list()
    input_image_width = input_shape[1]
    input_image_height = input_shape[2]
    input_n_channels = input_shape[3]
    
    with tf.name_scope(name):
        # convolution
        conv, w, b = conv2d(
            name,
            inputs,
            input_image_width, input_image_height, input_n_channels,
            kernal_size, kernal_size, output_depth,
            1, 1, 'SAME',
            True
        )
        
        # activation
        activation = activation(conv)
        
    # normalize
    with tf.name_scope("normalize_{}".format(name)):
        norm = tf.nn.local_response_normalization(activation)
        
    if more:
        return norm, activation, w, b
    else:
        return norm

def conv2d_pool(name, inputs, kernal_size, output_depth, activation=tf.nn.relu, poolling=tf.nn.max_pool, more=False):
    norm, a, w, b = conv2d_norm(name, inputs, kernal_size, output_depth, activation, more=True)

    # poolling
    with tf.name_scope("max_pool_{}".format(name)):
        pool = poolling(norm, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')
    
    if more:
        return pool, norm, a, w, b
    else:
        return pool

# will resize all images to the size of the first one
def cat_conv2d(name, inputs_list, kernal_size, output_depth, activation=tf.nn.relu, poolling=tf.nn.max_pool, more=False):
    # resize
    target_shape = inputs_list[0].get_shape().as_list()
    same_size_list = [inputs_list[0]]
    for i in range(1, len(inputs_list)):
        same_size_list.append(tf.image.resize_images(
            inputs_list[i], [target_shape[1], target_shape[2]], 1
        ))
        
    # concat
    inputs = tf.concat(same_size_list, 3)
    
    # convolute
    norm, a, w, b = conv2d_norm(name, inputs, kernal_size, output_depth, activation, more=True)

    if more:
        return norm, a, w, b
    else:
        return norm
    
def conv2d_one(name, inputs, activation=tf.nn.relu, more=False):
    # input shape
    input_shape = inputs.get_shape().as_list()
    input_image_width = input_shape[1]
    input_image_height = input_shape[2]
    input_n_channels = input_shape[3]
    
    with tf.name_scope(name):
        # convolution
        conv, w, b = conv2d(
            name,
            inputs,
            input_image_width, input_image_height, input_n_channels,
            input_image_width, input_image_height, input_n_channels,
            1, 1, 'VALID',
            True
        )
        
        resized = tf.reshape(conv, [-1, input_n_channels])
        
        # activation
        activation = tf.nn.l2_normalize(activation(resized), 1)
        
    if more:
        return activation, input_n_channels, w, b
    else:
        return activation, input_n_channels

def deconv2d(name, inputs, output_shape, kernal_size, stride=2, stddev=0.02, activation=tf.nn.relu, more=False):
    with tf.name_scope("deconv_{}".format(name)) as ns:
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable(
            'w_{}'.format(name),
            [kernal_size, kernal_size, output_shape[-1], inputs.get_shape()[-1]],
            initializer=tf.random_normal_initializer(stddev=stddev)
        )

        deconv = tf.nn.conv2d_transpose(
            inputs,
            w,
            output_shape=output_shape,
            strides=[1, stride, stride, 1]
        )

        biases = tf.get_variable('biases_{}'.format(name),
                                 [output_shape[-1]],
                                 initializer=tf.constant_initializer(0.0))

        deconv = tf.nn.bias_add(deconv, biases)
        activated = activation(deconv)
        norm = tf.nn.local_response_normalization(activated)

    if more:
        return norm, activated, w, biases
    else:
        return norm
    