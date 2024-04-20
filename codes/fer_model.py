from keras.models import *
from modules import *

def net_Inception(rank,category_n):
    input = Input(shape=(48, 48, 1))
    x = Inception(32, rank, input)
    x = Pooling(32, x)

    x = Inception(64, rank, x)
    x = Pooling(64, x)

    x = Inception(96, rank, x)
    x = Pooling(96, x)

    x = Inception(128, rank, x)
    x = Pooling(128, x)

    x = Activation('linear', name='end')(x)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    x = Dense(category_n)(x)
    x = Activation('softmax')(x)

    model = Model(inputs=input, outputs=x)
    return model


def net_Eception(rank,category_n):
    input = Input(shape=(48, 48, 1))
    x = Eception(32, rank, input)
    x = Pooling(32, x)

    x = Eception(64, rank, x)
    x = Pooling(64, x)

    x = Eception(96, rank, x)
    x = Pooling(96, x)

    x = Eception(128, rank, x)
    x = Pooling(128, x)

    x = Activation('linear', name='end')(x)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    x = Dense(category_n)(x)
    x = Activation('softmax')(x)

    model = Model(inputs=input, outputs=x)
    return model


def net_Lception(rank,category_n):
    input = Input(shape=(48, 48, 1))
    x = Lception(32, rank, input)
    x = Pooling(32, x)

    x = Lception(64, rank, x)
    x = Pooling(64, x)

    x = Lception(96, rank, x)
    x = Pooling(96, x)

    x = Lception(128, rank, x)
    x = Pooling(128, x)

    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    x = Dense(category_n)(x)
    x = Activation('softmax')(x)

    model = Model(inputs=input, outputs=x)
    return model