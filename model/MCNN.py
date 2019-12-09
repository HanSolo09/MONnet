import keras
from keras.initializers import glorot_uniform
from keras.layers import Dense, Conv2D, BatchNormalization, Activation, concatenate, Flatten, Input
from keras.regularizers import l2
from keras.models import Model
from keras.optimizers import Adam


def resnet_layer(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True):
    """2D Convolution-Batch Normalization-Activation stack builder
    # Arguments
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv2D number of filters
        kernel_size (int): Conv2D square kernel dimensions
        strides (int): Conv2D square stride dimensions
        activation (string): activation name
        batch_normalization (bool): whether to include batch normalization
        conv_first (bool): conv-bn-activation (True) or
            bn-activation-conv (False)
    # Returns
        x (tensor): tensor as input to the next layer
    """
    conv = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer=glorot_uniform(seed=0),
                  kernel_regularizer=l2(1e-4))

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x


def res_block(inputs,
              num_filters):
    x = resnet_layer(inputs=inputs,
                     num_filters=num_filters)
    x = resnet_layer(inputs=x,
                     num_filters=num_filters,
                     activation=None)
    x = keras.layers.add([inputs, x])
    x = Activation('relu')(x)

    return x


def conv_block(inputs,
               num_filters,
               strides=2):
    x = resnet_layer(inputs=inputs,
                     strides=strides,
                     num_filters=num_filters)
    y = resnet_layer(inputs=x,
                     num_filters=num_filters,
                     activation=None)
    x = resnet_layer(inputs=inputs,
                     num_filters=num_filters,
                     kernel_size=1,
                     strides=strides,
                     activation=None,
                     batch_normalization=False)
    x = keras.layers.add([x, y])
    x = Activation('relu')(x)

    return x


def multiscale_resnet(inputs, scale):
    if scale == 0:
        inputs = resnet_layer(inputs=inputs,
                              num_filters=16)

        # stage 0
        x = conv_block(inputs, 16)
        x = res_block(x, 16)
        x = res_block(x, 16)

    elif scale == 1:
        inputs = resnet_layer(inputs=inputs,
                              num_filters=8)

        # stage 0
        x = conv_block(inputs, 8)
        x = res_block(x, 8)
        x = res_block(x, 8)

        # stage 1
        x = conv_block(x, 16)
        x = res_block(x, 16)
        x = res_block(x, 16)

    elif scale == 2:
        inputs = resnet_layer(inputs=inputs,
                              num_filters=4)

        # stage 0
        x = res_block(inputs, 4)
        x = res_block(x, 4)
        x = res_block(x, 4)

        # stage 1
        x = conv_block(x, 8)
        x = res_block(x, 8)
        x = res_block(x, 8)

        # stage 2
        x = conv_block(x, 16, 3)
        x = res_block(x, 16)
        x = res_block(x, 16)

    return x


def MCNN(shape0, shape1, shape2, n_label):
    input0 = Input(shape=shape0, name='input0')
    input1 = Input(shape=shape1, name='input1')
    input2 = Input(shape=shape2, name='input2')

    x0 = multiscale_resnet(input0, scale=0)
    x1 = multiscale_resnet(input1, scale=1)
    x2 = multiscale_resnet(input2, scale=2)
    merged_feature = concatenate([x0, x1, x2], axis=-1)
    # pooling = MaxPooling2D(pool_size=(8, 8))(merged_feature)
    flatten = Flatten()(merged_feature)
    softmax_linear = Dense(n_label, activation='softmax', kernel_initializer=glorot_uniform(seed=0))(flatten)

    model = Model(inputs=[input0, input1, input2], outputs=softmax_linear)
    model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy',
                  metrics=['binary_crossentropy', 'accuracy'])

    # model.summary()
    # keras.utils.plot_model(model, to_file='MCNN.png', show_shapes=True)

    return model
