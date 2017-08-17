"""
This file contains the ResNet model, both single-input and multi-input

Original Source : https://github.com/raghakot/keras-resnet

Additions by : Ali Shelton
"""


from __future__ import division

import six
from keras.models import Model
from keras.layers import (
    Input,
    Activation,
    Dense,
    Flatten, 
    Dropout
)
from keras.layers.convolutional import (
    Conv2D,
    MaxPooling2D,
    AveragePooling2D
)
from keras.layers.merge import (
	add,
	concatenate
)
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras import backend as K


def _bn_relu(input):
    """Helper to build a BN -> relu block
    """
    norm = BatchNormalization(axis=CHANNEL_AXIS)(input)
    return Activation("relu")(norm)


def _conv_bn_relu(**conv_params):
    """Helper to build a conv -> BN -> relu block
    """
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1, 1))
    kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(1.e-4))

    def f(input):
        conv = Conv2D(filters=filters, kernel_size=kernel_size,
                      strides=strides, padding=padding,
                      kernel_initializer=kernel_initializer,
                      kernel_regularizer=kernel_regularizer)(input)
        return _bn_relu(conv)

    return f


def _bn_relu_conv(**conv_params):
    """Helper to build a BN -> relu -> conv block.
    This is an improved scheme proposed in http://arxiv.org/pdf/1603.05027v2.pdf
    """
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1, 1))
    kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(1.e-4))

    def f(input):
        activation = _bn_relu(input)
        return Conv2D(filters=filters, kernel_size=kernel_size,
                      strides=strides, padding=padding,
                      kernel_initializer=kernel_initializer,
                      kernel_regularizer=kernel_regularizer)(activation)

    return f


def _shortcut(input, residual):
    """Adds a shortcut between input and residual block and merges them with "sum"
    """
    # Expand channels of shortcut to match residual.
    # Stride appropriately to match residual (width, height)
    # Should be int if network architecture is correctly configured.
    input_shape = K.int_shape(input)
    residual_shape = K.int_shape(residual)
    stride_width = int(round(input_shape[ROW_AXIS] / residual_shape[ROW_AXIS]))
    stride_height = int(round(input_shape[COL_AXIS] / residual_shape[COL_AXIS]))
    equal_channels = input_shape[CHANNEL_AXIS] == residual_shape[CHANNEL_AXIS]

    shortcut = input
    # 1 X 1 conv if shape is different. Else identity.
    if stride_width > 1 or stride_height > 1 or not equal_channels:
        shortcut = Conv2D(filters=residual_shape[CHANNEL_AXIS],
                          kernel_size=(1, 1),
                          strides=(stride_width, stride_height),
                          padding="valid",
                          kernel_initializer="he_normal",
                          kernel_regularizer=l2(0.0001))(input)

    return add([shortcut, residual])

def _combine(input, residual):
    input_shape = K.int_shape(input)
    residual_shape = K.int_shape(residual)

    # 1 X 1 conv if shape is different. Else identity.
    shortcut = Conv2D(filters=residual_shape[CHANNEL_AXIS],
                      kernel_size=(3, 2),
                      strides=(4, 1),
                      padding="valid",
                      kernel_initializer="he_normal",
                      kernel_regularizer=l2(0.0001))(input)

    new_shape = K.int_shape(shortcut)

    altered = Conv2D(filters=residual_shape[CHANNEL_AXIS],
                      kernel_size=(1, 1),
                      strides=(2, 2),
                      padding="valid",
                      kernel_initializer="he_normal",
                      kernel_regularizer=l2(0.0001))(residual) 

    return add([shortcut, altered])

def _residual_block(block_function, filters, repetitions, is_first_layer=False):
    """Builds a residual block with repeating bottleneck blocks.
    """
    def f(input):
        for i in range(repetitions):
            init_strides = (1, 1)
            if i == 0 and not is_first_layer:
                init_strides = (2, 2)
            input = block_function(filters=filters, init_strides=init_strides,
                                   is_first_block_of_first_layer=(is_first_layer and i == 0))(input)
        return input

    return f


def basic_block(filters, init_strides=(1, 1), is_first_block_of_first_layer=False):
    """Basic 3 X 3 convolution blocks for use on resnets with layers <= 34.
    Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf
    """
    def f(input):

        if is_first_block_of_first_layer:
            # don't repeat bn->relu since we just did bn->relu->maxpool
            conv1 = Conv2D(filters=filters, kernel_size=(3, 3),
                           strides=init_strides,
                           padding="same",
                           kernel_initializer="he_normal",
                           kernel_regularizer=l2(1e-4))(input)
        else:
            conv1 = _bn_relu_conv(filters=filters, kernel_size=(3, 3),
                                  strides=init_strides)(input)

        residual = _bn_relu_conv(filters=filters, kernel_size=(3, 3))(conv1)
        return _shortcut(input, residual)

    return f


def bottleneck(filters, init_strides=(1, 1), is_first_block_of_first_layer=False):
    """Bottleneck architecture for > 34 layer resnet.
    Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf

    Returns:
        A final conv layer of filters * 4
    """
    def f(input):

        if is_first_block_of_first_layer:
            # don't repeat bn->relu since we just did bn->relu->maxpool
            conv_1_1 = Conv2D(filters=filters, kernel_size=(1, 1),
                              strides=init_strides,
                              padding="same",
                              kernel_initializer="he_normal",
                              kernel_regularizer=l2(1e-4))(input)
        else:
            conv_1_1 = _bn_relu_conv(filters=filters, kernel_size=(3, 3),
                                     strides=init_strides)(input)

        conv_3_3 = _bn_relu_conv(filters=filters, kernel_size=(3, 3))(conv_1_1)
        residual = _bn_relu_conv(filters=filters * 4, kernel_size=(1, 1))(conv_3_3)
        return _shortcut(input, residual)

    return f


def _handle_dim_ordering():
    global ROW_AXIS
    global COL_AXIS
    global CHANNEL_AXIS
    if K.image_dim_ordering() == 'tf':
        ROW_AXIS = 1
        COL_AXIS = 2
        CHANNEL_AXIS = 3
    else:
        CHANNEL_AXIS = 1
        ROW_AXIS = 2
        COL_AXIS = 3


def _get_block(identifier):
    if isinstance(identifier, six.string_types):
        res = globals().get(identifier)
        if not res:
            raise ValueError('Invalid {}'.format(identifier))
        return res
    return identifier


class ResnetBuilder(object):
    @staticmethod
    def build(input_shape, num_outputs, block_fn, repetitions):
        """Builds a custom ResNet like architecture.

        Args:
            input_shape: The input shape in the form (nb_channels, nb_rows, nb_cols)
            num_outputs: The number of outputs at final softmax layer
            block_fn: The block function to use. This is either `basic_block` or `bottleneck`.
                The original paper used basic_block for layers < 50
            repetitions: Number of repetitions of various block units.
                At each block unit, the number of filters are doubled and the input size is halved

        Returns:
            The keras `Model`.
        """
        _handle_dim_ordering()
        if len(input_shape) != 3:
            raise Exception("Input shape should be a tuple (nb_channels, nb_rows, nb_cols)")

        # Permute dimension order if necessary
        if K.image_dim_ordering() == 'tf':
            input_shape = (input_shape[1], input_shape[2], input_shape[0])

        # Load function from str if needed.
        block_fn = _get_block(block_fn)

        input = Input(shape=input_shape)
        conv1 = _conv_bn_relu(filters=64, kernel_size=(7, 7), strides=(2, 2))(input)
        pool1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")(conv1)

        block = pool1
        filters = 64
        for i, r in enumerate(repetitions):
            block = _residual_block(block_fn, filters=filters, repetitions=r, is_first_layer=(i == 0))(block)
            filters *= 2

        # Last activation
        block = _bn_relu(block)

        # Classifier block
        block_shape = K.int_shape(block)
        pool2 = AveragePooling2D(pool_size=(block_shape[ROW_AXIS], block_shape[COL_AXIS]),
                                 strides=(1, 1))(block)
        flatten1 = Flatten()(pool2)
        dense = Dense(units=num_outputs, kernel_initializer="he_normal",
                      activation="linear")(flatten1)

        model = Model(inputs=input, outputs=dense)
        return model

    @staticmethod
    def build_multi_input(input_shapes, num_outputs, block_fn, repetitions):
        """Builds a custom ResNet like architecture.

        Args:
            input_shapes: The input shapes of multiple inputs in the form [(nb_channels0, nb_rows0, nb_cols0), ...]
            num_outputs: The number of outputs at final softmax layer
            block_fn: The block function to use. This is either `basic_block` or `bottleneck`.
                The original paper used basic_block for layers < 50
            repetitions: Number of repetitions of various block units.
                At each block unit, the number of filters are doubled and the input size is halved

        Returns:
            The keras `Model`.
        """

        _handle_dim_ordering()
        for i, input_shape in enumerate(input_shapes[0:1]):      	
            if len(input_shape) != 3:
                raise Exception("Input shape should be a tuple (nb_channels, nb_rows, nb_cols)")

            # Permute dimension order if necessary
            if K.image_dim_ordering() == 'tf':
                input_shapes[i] = (input_shape[1], input_shape[2], input_shape[0])

        # Load function from str if needed.
        block_fn = _get_block(block_fn)

        inputs = [Input(shape=input_shape) for input_shape in input_shapes]
        """
        convs1 = [_conv_bn_relu(filters=64, kernel_size=(7, 7), strides=(2, 2))(input) for input in inputs]
        """
        
        # demographic input mini-net
        convs1 = [_conv_bn_relu(filters=64, kernel_size=(7, 7), strides=(2, 2))(input) for input in inputs[0:1]]

        dense_demo = [Dense(units=64, kernel_initializer="he_normal", activation="relu")(input) for input in inputs[1:]]
        dense_demo = [Dense(units=32, kernel_initializer="he_normal", activation="relu")(demo) for demo in dense_demo]
        

        pools1 = [MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")(conv1) for conv1 in convs1]

        blocks = pools1
        filters = 64
        for i, r in enumerate(repetitions):
            blocks = [_residual_block(block_fn, filters=filters, repetitions=r, is_first_layer=(i == 0))(block) for block in blocks]
            filters *= 2

        # Last activation
        blocks = [_bn_relu(block) for block in blocks]

        # Classifier block
        block_shapes = [K.int_shape(block) for block in blocks]
        pool2 = [AveragePooling2D(pool_size=(block_shape[ROW_AXIS], block_shape[COL_AXIS]),
                                 strides=(1, 1))(block) for block_shape, block in zip(block_shapes, blocks)]
        flattened = [Flatten()(pool) for pool in pool2]
        pre_dense = [Dense(units=128, kernel_initializer="he_normal", 
        	          activation="linear")(flat) for flat in flattened]


        combined_dense = pre_dense[0]
       	for dense in pre_dense[1:]:
            combined_dense = add([combined_dense, dense])

        dense1 = Dense(units=64, kernel_initializer="he_normal",
                      activation="relu")(combined_dense)
        """
        dense2 = Dense(units=num_outputs, kernel_initializer="he_normal",
                      activation="linear")(dense1)

        """
        # Add demographic info before decision
        dense2 = Dense(units=32, kernel_initializer="he_normal",
                      activation="relu")(dense1)

       	for demo in dense_demo:
            dense2 = concatenate([dense2, demo], axis=-1)
        dense2 = Dense(units=16, kernel_initializer="he_normal", \
        	activation="relu")(dense2)
        dense2 = Dense(units=8, kernel_initializer="he_normal", \
        	activation="relu")(dense2)
        dense2 = Dense(units=num_outputs, kernel_initializer="he_normal", \
        	activation="linear")(dense2)

        model = Model(inputs=inputs, outputs=dense2)
        return model

    @staticmethod
    def build_resnet_18(input_shape, num_outputs):
        return ResnetBuilder.build(input_shape, num_outputs, basic_block, [2, 2, 2, 2])

    @staticmethod
    def build_resnet_34(input_shape, num_outputs):
        return ResnetBuilder.build(input_shape, num_outputs, basic_block, [3, 4, 6, 3])

    @staticmethod
    def build_resnet_50(input_shape, num_outputs):
        return ResnetBuilder.build(input_shape, num_outputs, bottleneck, [3, 4, 6, 3])

    @staticmethod
    def build_resnet_101(input_shape, num_outputs):
        return ResnetBuilder.build(input_shape, num_outputs, bottleneck, [3, 4, 23, 3])

    @staticmethod
    def build_resnet_152(input_shape, num_outputs):
        return ResnetBuilder.build(input_shape, num_outputs, bottleneck, [3, 8, 36, 3])

    @staticmethod
    def build_resnet_18_multi(input_shapes, num_outputs):
    	return ResnetBuilder.build_multi_input(input_shapes, num_outputs, basic_block, [2, 2, 2, 2])