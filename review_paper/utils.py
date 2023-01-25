from tensorflow.keras.preprocessing.image import ImageDataGenerator, DirectoryIterator
from tensorflow.keras.applications.resnet import preprocess_input as resnet_preprocessing
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping
from tensorflow.keras.applications import ResNet50, EfficientNetB0, InceptionResNetV2, InceptionV3, MobileNet, \
    MobileNetV2, NASNetMobile, ResNet50V2, VGG16, Xception
from tensorflow.keras.layers import GlobalAveragePooling2D, BatchNormalization, Dropout, Dense, Flatten, Reshape, Conv2D

import numpy as np
import pandas as pd
import os
from pathlib import Path


def calc_class_weights(train_iterator):
    """
    Calculate class weighs dictionary to use as input for the cnn training. This is useful if the training set is
    imbalanced.

    The weight of class "i" is calculated as the number of samples in the most populated class divided by the number of
    samples in class i (max_class_frequency / class_frequency).
    Note that the class weights are capped at 10. This is done in order to avoid placing too much weight on
    small fraction of the dataset. For the same reason, the weight is set to 1 for any class in the training set that
    contains fewer than 5 samples.

    :param class_counts: A list with the number of files for each class.
    :return:
    """

    # Fixed parameters
    class_counts = np.unique(train_iterator.classes, return_counts=True)
    max_freq = max(class_counts[1])
    class_weights = [max_freq / count for count in class_counts[1]]

    print("Classes: " + str(class_counts[0]))
    print("Samples per class: " + str(class_counts[1]))
    print("Class weights: " + str(class_weights))

    return class_weights


def unfreeze_layers(model, last_fixed_layer):
    # Retrieve the index of the last fixed layer and add 1 so that it is also set to not trainable
    try:
        first_trainable = model.layers.index(model.get_layer(last_fixed_layer)) + 1
    except:
        model.summary()
    # Set which layers are trainable.
    for layer_idx, layer in enumerate(model.layers):
        if not isinstance(layer, BatchNormalization):
            layer.trainable = layer_idx >= first_trainable
    return model


def train_model(model, model_base_name, rotation, shear, zoom, brightness, lr, last_fixed_layer, batch_size,
                preprocessing_function, base_path, data_path, image_size, name_suffix='', epochs=None):
    model_name = f'{model_base_name}_r{rotation}_s{shear}_z{zoom}_b{brightness}_lr{lr}_l{last_fixed_layer}{name_suffix}'
    if os.path.exists(Path(base_path) / (model_name + '_preds.csv')):
        print(f'{model_name} already trained')
        return model_name
    print(f'Now training {model_name}')

    train_generator = ImageDataGenerator(
        horizontal_flip=True,
        vertical_flip=True,
        rotation_range=rotation,
        shear_range=shear,
        zoom_range=zoom,
        brightness_range=brightness,
        fill_mode='nearest',
        preprocessing_function=preprocessing_function,
    )
    train_iterator = train_generator.flow_from_directory(
        Path(data_path) / 'train',
        target_size=image_size,
        class_mode='categorical',
        batch_size=batch_size,
        follow_links=True,
        interpolation='bilinear',
    )

    valid_generator = ImageDataGenerator(
        fill_mode='nearest',
        preprocessing_function=resnet_preprocessing
    )
    valid_iterator = valid_generator.flow_from_directory(
        Path(data_path) / 'valid',
        batch_size=batch_size,
        target_size=image_size,
        class_mode='categorical',
        follow_links=True,
        interpolation='bilinear',
    )

    loss_weights = calc_class_weights(train_iterator)

    optimiser = Adam(lr=lr)
    model.compile(
        optimizer=optimiser,
        loss='categorical_crossentropy',
        metrics=['accuracy'],
        loss_weights=loss_weights,
    )

    callbacks = [CSVLogger(Path(base_path) / (model_name + '.csv'))]
    if not epochs:
         callbacks.append(EarlyStopping(monitor='val_loss', min_delta=0.5, patience=25, verbose=1, mode='auto',
                                   restore_best_weights=True))
    
    
    model.fit(
        x=train_iterator,
        batch_size=batch_size,
        epochs=epochs,
        verbose=True,
        validation_data=valid_iterator,
        class_weight=dict(zip(range(6), loss_weights)),
        workers=8,
        callbacks=callbacks
    )
    model.save(Path(base_path) / (model_name + '.h5'))
    return model_name


def get_resnet_model(image_size):
    base_model = ResNet50(include_top=False, weights='imagenet', input_shape=(image_size[0], image_size[1], 3))
    top_model = Flatten()(base_model.output)
    top_model = Dense(6, activation='softmax', name='diagnosis')(top_model)
    return Model(inputs=base_model.input, outputs=top_model, name="ResNet50")


def get_efficientnet_model(image_size):
    model = EfficientNetB0(include_top=False, input_shape=(image_size[0], image_size[1], 3), weights="imagenet")

    # Rebuild top
    x = GlobalAveragePooling2D(name="avg_pool")(model.output)
    x = BatchNormalization()(x)

    top_dropout_rate = 0.2
    x = Dropout(top_dropout_rate, name="top_dropout")(x)
    outputs = Dense(6, activation="softmax", name="pred")(x)

    return Model(model.input, outputs, name="EfficientNetB0")


def get_inceptionresnet_model(image_size):
    base_model = InceptionResNetV2(include_top=False, weights='imagenet', input_shape=(image_size[0], image_size[1], 3))
    top_model = GlobalAveragePooling2D(name="avg_pool")(base_model.output)
    top_model = Dense(6, activation='softmax', name='diagnosis')(top_model)
    return Model(inputs=base_model.input, outputs=top_model, name="InceptionResNetV2")


def get_inception_model(image_size):
    base_model = InceptionV3(include_top=False, weights='imagenet', input_shape=(image_size[0], image_size[1], 3))
    top_model = GlobalAveragePooling2D(name="avg_pool")(base_model.output)
    top_model = Dense(6, activation='softmax', name='diagnosis')(top_model)
    return Model(inputs=base_model.input, outputs=top_model, name="InceptionV3")


def get_mobilenetv1_model(image_size):
    base_model = MobileNet(include_top=False, weights='imagenet', input_shape=(image_size[0], image_size[1], 3))
    top_model = GlobalAveragePooling2D(name="avg_pool")(base_model.output)
    top_model = Reshape((1, 1, 1024), name='reshape_1')(top_model)
    top_model = Dropout(0.2, name='dropout')(top_model)
    top_model = Conv2D(6, (1, 1), padding='same', name='conv_preds')(top_model)
    top_model = Reshape((6,), name='reshape_2')(top_model)
    top_model = Dense(6, activation='softmax', name='diagnosis')(top_model)
    return Model(inputs=base_model.input, outputs=top_model, name="MobileNetV1")


def get_mobilenetv2_model(image_size):
    base_model = MobileNetV2(include_top=False, weights='imagenet', input_shape=(image_size[0], image_size[1], 3))
    top_model = GlobalAveragePooling2D(name="avg_pool")(base_model.output)
    top_model = Dense(6, activation='softmax', name='diagnosis')(top_model)
    return Model(inputs=base_model.input, outputs=top_model, name="MobileNetV2")


def get_nasnetmobile_model(image_size):
    base_model = NASNetMobile(include_top=False, weights='imagenet', input_shape=(image_size[0], image_size[1], 3))
    top_model = GlobalAveragePooling2D(name="avg_pool")(base_model.output)
    top_model = Dense(6, activation='softmax', name='diagnosis')(top_model)
    return Model(inputs=base_model.input, outputs=top_model, name="NASNetMobile")


def get_resnetv2_model(image_size):
    base_model = ResNet50V2(include_top=False, weights='imagenet', input_shape=(image_size[0], image_size[1], 3))
    top_model = GlobalAveragePooling2D(name="avg_pool")(base_model.output)
    top_model = Dense(6, activation='softmax', name='diagnosis')(top_model)
    return Model(inputs=base_model.input, outputs=top_model, name="ResNetV2")


def get_vgg_model(image_size):
    base_model = VGG16(include_top=False, weights='imagenet', input_shape=(image_size[0], image_size[1], 3))
    top_model = Flatten()(base_model.output)
    #top_model = Dense(4096, activation='relu')(top_model)
    #top_model = Dense(4096, activation='relu')(top_model)
    top_model = Dense(6, activation='softmax', name='diagnosis')(top_model)
    return Model(inputs=base_model.input, outputs=top_model, name="VGG")


def get_xception_model(image_size):
    base_model = Xception(include_top=False, weights='imagenet', input_shape=(image_size[0], image_size[1], 3))
    top_model = GlobalAveragePooling2D(name="avg_pool")(base_model.output)
    top_model = Dense(6, activation='softmax', name='diagnosis')(top_model)
    return Model(inputs=base_model.input, outputs=top_model, name="Xception")


def validate_model(base_path, model_name, preprocessing_function, data_path, image_size, remove_weights=True, preds_path=None):
    if not preds_path:
        preds_path = Path(base_path) / (model_name + '_preds.csv')
    print(preds_path)
    if os.path.exists(preds_path):
        print(f'{model_name} already validated')
        return model_name

    print('Now validating', model_name)
    valid_generator = ImageDataGenerator(
        fill_mode='nearest',
        preprocessing_function=preprocessing_function
    )
    valid_iterator = valid_generator.flow_from_directory(
        Path(data_path) / 'valid',
        batch_size=8,
        target_size=image_size,
        class_mode='categorical',
        follow_links=True,
        interpolation='bilinear',
        shuffle=False
    )

    model = load_model(Path(base_path) / (model_name + '.h5'))
    preds = [np.argmax(pred) for pred in model.predict(valid_iterator)]
    actual = valid_iterator.labels
    pd.DataFrame.from_dict({'actual': actual, 'pred': preds}).to_csv(preds_path)
    if remove_weights:
        os.remove(Path(base_path) / (model_name + '.h5'))

        
def test_model(base_path, model_name, preprocessing_function, data_path, image_size, preds_path=None):
    if not preds_path:
        preds_path = Path(base_path) / (model_name + '_preds.csv')
    print(preds_path)
#     if os.path.exists(preds_path):
#         print(f'{model_name} already validated')
#         return model_name

    print('Now validating', model_name)
    valid_generator = ImageDataGenerator(
        fill_mode='nearest',
        preprocessing_function=preprocessing_function
    )
    valid_iterator = valid_generator.flow_from_directory(
        Path(data_path) / 'valid',
        batch_size=8,
        target_size=image_size,
        class_mode='categorical',
        follow_links=True,
        interpolation='bilinear',
        shuffle=False
    )

    model = load_model(Path(base_path) / (model_name + '.h5'))
    preds = [np.argmax(pred) for pred in model.predict(valid_iterator)]
    actual = valid_iterator.labels
    filenames = valid_iterator.filenames
    classes = {value: key for key, value in valid_iterator.class_indices.items()}
    pred_classes = [classes[pred] for pred in preds]
    actual_classes = [classes[dx] for dx in actual]
    pd.DataFrame.from_dict({
        'filename': filenames, 
        'actual': actual, 
        'pred': preds, 
        'actual_class': actual_classes, 
        'pred_class': pred_classes
    }).to_csv(preds_path)
    