import tensorflow as tf
from keras.utils import get_custom_objects
from keras.layers    import *
from keras import regularizers
from keras.layers import Activation
tf.random.set_seed(47)


def swish(inputs):
    return (K.sigmoid(inputs) * inputs)
def h_swish(inputs):
    return inputs * tf.nn.relu6(inputs + 3) / 6
    
get_custom_objects().update({'swish': Activation(swish)})
get_custom_objects().update({'h_swish': Activation(h_swish)})

def DWConv(size,x):
    x = DepthwiseConv2D(kernel_size=(5,5), strides=(1, 1), padding='same',kernel_initializer="he_uniform",
                                           depth_multiplier=1, depthwise_regularizer=regularizers.l2(0.01))(x)
    x = BatchNormalization()(x)
    x = Activation('h_swish')(x)
    return x

def Conv(size,x):
    x = Conv2D(size, (3,3) , strides=(1,1), padding='same',kernel_initializer="he_uniform",
                                                                  kernel_regularizer=regularizers.l2(0.01))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def Conv_1(size,x):
    x = Conv2D(size, (1,1) , strides=(1,1), padding='same',kernel_initializer="he_uniform",
                                                                   kernel_regularizer=regularizers.l2(0.01))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def pool(size,x):
    x = MaxPooling2D((3,3),strides=(1,1),padding='same')(x)
    x = Conv_1(size,x)
    return x

def Pooling(size,x):
    x = MaxPooling2D((2,2))(x)
    return x

