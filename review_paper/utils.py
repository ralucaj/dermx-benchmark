from tensorflow.keras.preprocessing.image import ImageDataGenerator, DirectoryIterator
from tensorflow.keras.applications.resnet import preprocess_input as resnet_preprocessing
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping
from tensorflow.keras.applications import ResNet50, EfficientNetB0
from tensorflow.keras.layers import GlobalAveragePooling2D, BatchNormalization, Dropout, Dense, Flatten

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
    class_weights = []
    max_freq = max(class_counts[1])
    class_weights = [max_freq / count for count in class_counts[1]]

    print("Classes: " + str(class_counts[0]))
    print("Samples per class: " + str(class_counts[1]))
    print("Class weights: " + str(class_weights))

    return class_weights


def unfreeze_layers(model, last_fixed_layer):
    # Retrieve the index of the last fixed layer and add 1 so that it is also set to not trainable
    first_trainable = model.layers.index(model.get_layer(last_fixed_layer)) + 1

    # Set which layers are trainable.
    for layer_idx, layer in enumerate(model.layers):
        if not isinstance(layer, BatchNormalization):
            layer.trainable = layer_idx >= first_trainable
    return model


def train_model(model, model_base_name, rotation, shear, zoom, brightness, lr, last_fixed_layer, batch_size, preprocessing_function, base_path, data_path):
    model_name = f'{model_base_name}_r{rotation}_s{shear}_z{zoom}_b{brightness}_lr{lr}_l{last_fixed_layer}'
    if os.path.exists(Path(base_path) / (model_name + '.h5')):
        print(f'{model_name} already trained')
        return
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
        target_size=(400, 300),
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
        target_size=(400, 300),
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

    logger = CSVLogger(Path(base_path) / (model_name + '.csv'))
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.5, patience=5, verbose=1, mode='auto',
                                   restore_best_weights=True)

    model.fit(
        x=train_iterator,
        batch_size=batch_size,
        epochs=100,
        verbose=True,
        validation_data=valid_iterator,
        class_weight=dict(zip(range(6), loss_weights)),
        workers=8,
        callbacks=[logger, early_stopping]
    )
    model.save(Path(base_path) / (model_name + '.h5'))
    return model_name


def get_resnet_model():
    base_model = ResNet50(include_top=False, weights='imagenet', input_shape=(400, 300, 3))
    top_model = Flatten()(base_model.output)
    top_model = Dense(6, activation='softmax', name='diagnosis')(top_model)
    return Model(inputs=base_model.input, outputs=top_model, name="ResNet50")


def get_efficientnet_model():
    model = EfficientNetB0(include_top=False, input_shape=(400, 300, 3), weights="imagenet")

    # Rebuild top
    x = GlobalAveragePooling2D(name="avg_pool")(model.output)
    x = BatchNormalization()(x)

    top_dropout_rate = 0.2
    x = Dropout(top_dropout_rate, name="top_dropout")(x)
    outputs = Dense(6, activation="softmax", name="pred")(x)

    return Model(model.input, outputs, name="EfficientNetB0")


def validate_model(base_path, model_name, preprocessing_function, data_path):
    if os.path.exists(Path(base_path) / (model_name + '_preds.csv')):
        print(f'{model_name} already validated')
        return

    print('Now validating', model_name)
    valid_generator = ImageDataGenerator(
        fill_mode='nearest',
        preprocessing_function=preprocessing_function
    )
    valid_iterator = valid_generator.flow_from_directory(
        Path(data_path) / 'valid',
        batch_size=8,
        target_size=(400, 300),
        class_mode='categorical',
        follow_links=True,
        interpolation='bilinear',
        shuffle=False
    )

    model = load_model(Path(base_path) / (model_name + '.h5'))
    preds = [np.argmax(pred) for pred in model.predict(valid_iterator)]
    actual = valid_iterator.labels
    pd.DataFrame.from_dict({'actual': actual, 'pred': preds}).to_csv(Path(base_path) / (model_name + '_preds.csv'))
