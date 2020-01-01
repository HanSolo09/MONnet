from model.MCNN import *


def SingleCNN(shape, n_label,scale):
    input = Input(shape=shape, name='input')
    x = multiscale_resnet(input, scale=scale)
    flatten = Flatten()(x)
    softmax_linear = Dense(n_label, activation='softmax', kernel_initializer=glorot_uniform(seed=0))(flatten)

    model = Model(inputs=input, outputs=softmax_linear)
    model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy',
                  metrics=['binary_crossentropy', 'accuracy'])

    # model.summary()
    # keras.utils.plot_model(model, to_file='SingleCNN.png', show_shapes=True)

    return model
