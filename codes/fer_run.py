import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import *
from fer_model import *

img_width, img_height = 48, 48
train_data_dir = 'fer2013plus/train'           #Please change to your own dataset address
validation_data_dir = 'fer2013plus/valid'
test_data_dir = 'fer2013plus/test'

#Note modifications train_samples,validation_samples,test_samples when using the fer2013 dataset
train_samples = 25045
validation_samples = 3191
test_samples = 3137
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
val_datagen = ImageDataGenerator(rescale=1. / 255)
test_datagen = ImageDataGenerator(rescale=1. / 255)
train_generator = train_datagen.flow_from_directory(
                                                            train_data_dir,
                                                            target_size=(img_width, img_height),
                                                            batch_size=batch_size,
                                                            color_mode='grayscale',
                                                            class_mode='categorical')

valid_generator = val_datagen.flow_from_directory(
                                                            validation_data_dir,
                                                            target_size=(img_width, img_height),
                                                            batch_size=batch_size,
                                                            color_mode='grayscale',
                                                            class_mode='categorical')
test_generator = test_datagen.flow_from_directory(
                                                            test_data_dir,
                                                            target_size=(img_width, img_height),
                                                            batch_size=batch_size,
                                                            color_mode='grayscale',
                                                            class_mode='categorical')

labels = train_generator.class_indices

print(labels)

mode = 'train'             #Selectable operating modes include ’train‘ and 'test'
model_save_dir = 'FER_E4x.h5'        #Add the storage address of the trained model, if mode == 'train'
model_test_dir = 'FER_E8x.h5'        #Add the address of the model to be predicted, if mode == 'test'


if mode == 'train':
    model = net_Eception(4,8)  #Optional models:net_Inception,net_Eception,net_Lception.
    # Please note that setting the rank,net_Inception is allowed to set 1-6, others are allowed to set 1-8.
    # If the data set is fer2013,then category_n=7.If the data set is fer+,then category_n=8
    model.summary()
    model.compile(loss='categorical_crossentropy',
                  optimizer='sgd',
                  metrics=['accuracy'])
    def schedule(epoch):
        if epoch <= 40:
            lr = 0.01
        elif epoch <= 60:
            lr = 0.001
        elif epoch <= 80:
            lr = 0.0001
        return lr
    change_lr = LearningRateScheduler(schedule, verbose=1)
    model_checkpoint = ModelCheckpoint(model_save_dir, 'val_accuracy', verbose=1, save_best_only=True)
    history = model.fit_generator(
        train_generator,
        steps_per_epoch=train_samples // batch_size,
        epochs=80,
        verbose=1,
        validation_data=valid_generator,
        validation_steps=validation_samples // batch_size,
        callbacks=[change_lr, model_checkpoint])

elif mode == 'test':
    model = load_model(model_test_dir)
    model.summary()
    model.evaluate(test_generator)
    # !!! If the model to be tested is net_Lception,
    # run the following code and comment out the above three lines of code.
    # model = net_Lception(4,8)      # Note that setting the corresponding rank and category_n
    # model.load_weights(model_test_dir)
    # model.compile(loss='categorical_crossentropy',
    #               optimizer='sgd',
    #               metrics=['accuracy'])
    # model.evaluate(test_generator)
