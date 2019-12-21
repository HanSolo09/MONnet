import keras
from keras.optimizers import Adam
from utils.ssrn import ResnetBuilder


def SSRN(shape, n_label):
    """
    Dumb function to wrap SSRN model.
    """
    model = ResnetBuilder.build_resnet_8(shape, n_label)
    model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy',
                  metrics=['binary_crossentropy', 'accuracy'])

    # keras.utils.plot_model(model, to_file='SSRN.png', show_shapes=True)
    return model
