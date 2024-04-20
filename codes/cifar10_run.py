from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.callbacks import *
from cifar10_model import *

img_width, img_height = 96, 96
train_data_dir = 'cifar-10/train'    #Please change to your own cifar10 dataset address
test_data_dir = 'cifar-10/test'
train_samples = 50000
test_samples = 10000
batch_size = 32
train_datagen = ImageDataGenerator(
                                    rescale=1. / 255,
                                    rotation_range=30,
                                    width_shift_range=0.1,
                                    height_shift_range=0.1,
                                    shear_range=0.1,
                                    zoom_range=0.1,
                                    horizontal_flip=True,
                                    fill_mode='nearest'
                                    )
test_datagen = ImageDataGenerator(rescale=1. / 255)
train_generator = train_datagen.flow_from_directory(
                                                            train_data_dir,
                                                            target_size=(img_width, img_height),
                                                            batch_size=batch_size,
                                                            color_mode='rgb',
                                                            class_mode='categorical')

test_generator = test_datagen.flow_from_directory(
                                                            test_data_dir,
                                                            target_size=(img_width, img_height),
                                                            batch_size=batch_size,
                                                            color_mode='rgb',
                                                            class_mode='categorical')
labels = train_generator.class_indices

print(labels)

mode = 'train'                       #Selectable operating modes include ’train‘ and 'test'
model_save_dir = 'C10_E4x.h5'        #Add the storage address of the trained model, if mode == 'train'
model_test_dir = 'C10_E8x.h5'        #Add the address of the model to be predicted, if mode == 'test'

if mode == 'train':

    model = net_Eception(rank = 4)    #Optional models:net_Inception,net_Eception,net_Lception.
    # Please note that setting the rank,net_Inception is allowed to set 1-6, others are allowed to set 1-8.
    model.summary()
    sgd = optimizers.SGD(learning_rate=0.1, momentum=0.9, nesterov=False)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])
    def schedule(epoch):
        if epoch <= 10:
            lr = 0.001
        elif epoch <= 20:
            lr = 0.0001
        elif epoch <= 30:
            lr = 0.00001
        elif epoch <= 40:
            lr = 0.000001
        elif epoch <= 50:
            lr = 0.0000001
        return lr

    change_lr = LearningRateScheduler(schedule, verbose=1)
    history = model.fit_generator(
        train_generator,
        steps_per_epoch=train_samples // batch_size,
        epochs=50,
        verbose=1,
        callbacks=[change_lr])

    model.save(model_save_dir)

elif mode == 'test':
    model = load_model(model_test_dir)
    model.summary()
    model.evaluate(test_generator)
    # !!! If the model to be tested is net_Lception,
    # run the following code and comment out the above three lines of code.
    # model = net_Lception(rank = 4)     # Note that setting the corresponding rank
    # model.load_weights(model_test_dir)
    # model.compile(loss='categorical_crossentropy',
    #                metrics=['accuracy'])
    # model.evaluate(test_generator)