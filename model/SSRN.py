import keras
from utils.ssrn import ResnetBuilder


def SSRN(shape, n_label):
    """
    Dumb function to wrap SSRN model.
    """
    model = ResnetBuilder.build_resnet_8(shape, n_label)
    # keras.utils.plot_model(model, to_file='SSRN.png', show_shapes=True)
    return model
