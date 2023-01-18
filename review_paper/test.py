import tensorflow as tf
from utils import train_model, unfreeze_layers, get_resnet_model, get_efficientnet_model, validate_model, \
    get_xception_model, get_vgg_model, get_resnetv2_model, get_nasnetmobile_model, get_mobilenetv2_model, \
    get_mobilenetv1_model, get_inception_model, get_inceptionresnet_model
from tensorflow.keras.applications.resnet import preprocess_input as resnet_preprocessing
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input as inceptionresnet_preprocessing
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocessing
from tensorflow.keras.applications.inception_v3 import preprocess_input as inception_preprocessing
from tensorflow.keras.applications.mobilenet import preprocess_input as mobilenet_preprocessing
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenetv2_preprocessing
from tensorflow.keras.applications.nasnet import preprocess_input as nasnet_preprocessing
from tensorflow.keras.applications.resnet_v2 import preprocess_input as resnetv2_preprocessing
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocessing
from tensorflow.keras.applications.xception import preprocess_input as xception_preprocessing
from tensorflow.errors import ResourceExhaustedError
import os
import gc
from tensorflow.keras import backend as K
from pathlib import Path


models = [
    {
        'model_name': 'resnet50_r20_s0.5_z0.5_b[0.5, 1.5]_lr0.0001_lconv5_block3_out',
        'preprocessing_function': resnet_preprocessing,
        'rotation': 20,
        'shear': 0.5,
        'zoom': 0.5,
        'brightness': [0.5, 1.5],
        'lr': 0.0001,
        'last_fixed_layer': 'conv5_block3_out',
        'model': get_resnet_model((400, 300)),
        'base_path': "/home/ubuntu/hot-store/25e/resnet",
        'image_size': (400, 300),
        'epochs': 10,
    },
    {
        'model_name': 'efficientnetb0_r20_s0.25_z0.5_b[0.5, 1.5]_lr0.0001_lblock6d_add',
        'preprocessing_function': efficientnet_preprocessing,
        'rotation': 20,
        'shear': 0.25,
        'zoom': 0.5,
        'brightness': [0.5, 1.5],
        'lr': 0.0001,
        'last_fixed_layer': 'block6d_add',
        'model': get_efficientnet_model((400, 300)),
        'base_path': "/home/ubuntu/hot-store/25e/efficientnet",
        'image_size': (400, 300),
        'epochs': 10,
    },
    {
        'model_name': 'inceptionresnetv2_r20_s0.25_z0.5_b[0.75, 1.25]_lr0.0001_lblock8_9_ac',
        'preprocessing_function': inceptionresnet_preprocessing,
        'rotation': 20,
        'shear': 0.25,
        'zoom': 0.5,
        'brightness': [0.75, 1.25],
        'lr': 0.0001,
        'last_fixed_layer': 'block8_9_ac',
        'model': get_inceptionresnet_model((400, 300)),
        'base_path': "/home/ubuntu/hot-store/25e/inceptionresnet",
        'image_size': (400, 300),
        'epochs': 10,
    },
    {
        'model_name': 'inception_r20_s0.5_z0.5_b[0.5, 1.5]_lr0.001_lactivation_288',
        'preprocessing_function': inception_preprocessing,
        'rotation': 20,
        'shear': 0.5,
        'zoom': 0.5,
        'brightness': [0.5, 1.5],
        'lr': 0.001,
        'last_fixed_layer': 'activation_288',
        'model': get_inception_model((400, 300)),
        'base_path': "/home/ubuntu/hot-store/25e/inception",
        'image_size': (400, 300),
        'epochs': 10,
    },
    {
        'model_name': 'mobilenetv1_r10_s0.5_z0.5_b[0.5, 1.5]_lr0.0001_lconv_pw_12_relu',
        'preprocessing_function': mobilenet_preprocessing,
        'rotation': 10,
        'shear': 0.5,
        'zoom': 0.5,
        'brightness': [0.5, 1.5],
        'lr': 0.0001,
        'last_fixed_layer': 'conv_pw_12_relu',
        'model': get_mobilenetv1_model((400, 300)),
        'base_path': "/home/ubuntu/hot-store/25e/mobilenetv1",
        'image_size': (400, 300),
        'epochs': 10,
    },
    {
        'model_name': 'mobilenetv2_r10_s0.25_z0.5_b[0.5, 1.5]_lr0.0001_lblock_15_add',
        'preprocessing_function': mobilenetv2_preprocessing,
        'rotation': 10,
        'shear': 0.25,
        'zoom': 0.5,
        'brightness': [0.5, 1.5],
        'lr': 0.0001,
        'last_fixed_layer': 'block_15_add',
        'model': get_mobilenetv2_model((400, 300)),
        'base_path': "/home/ubuntu/hot-store/25e/mobilenetv2",
        'image_size': (400, 300),
        'epochs': 10,
    },
    {
        'model_name': 'nasnetmobile_r20_s0.25_z0.5_b[0.5, 1.0]_lr0.0001_lnormal_concat_11',
        'preprocessing_function': nasnet_preprocessing,
        'rotation': 20,
        'shear': 0.25,
        'zoom': 0.5,
        'brightness': [0.5, 1.0],
        'lr': 0.0001,
        'last_fixed_layer': 'normal_concat_11',
        'model': get_nasnetmobile_model((224, 224)),
        'base_path': "/home/ubuntu/hot-store/25e/nasnetmobile",
        'image_size': (224, 224),
        'epochs': 10,
    },
    {
        'model_name': 'resnetv2_r20_s0.25_z0.25_b[0.5, 1.0]_lr0.001_lpost_relu',
        'preprocessing_function': resnetv2_preprocessing,
        'rotation': 20,
        'shear': 0.25,
        'zoom': 0.25,
        'brightness': [0.5, 1.0],
        'lr': 0.001,
        'last_fixed_layer': 'post_relu',
        'model': get_resnetv2_model((400, 300)),
        'base_path': "/home/ubuntu/hot-store/25e/resnetv2",
        'image_size': (400, 300),
        'epochs': 10,
    },
    {
        'model_name': 'vgg_r10_s0.0_z0.25_b[0.5, 1.0]_lr0.01_lblock5_pool',
        'preprocessing_function': vgg_preprocessing,
        'rotation': 10,
        'shear': 0.0,
        'zoom': 0.25,
        'brightness': [0.5, 1.0],
        'lr': 0.01,
        'last_fixed_layer': 'block5_pool',
        'model': get_vgg_model((400, 300)),
        'base_path': "/home/ubuntu/hot-store/25e/vgg",
        'image_size': (400, 300),
        'epochs': 10,
    },
    {
        'model_name': 'xception_r10_s0.25_z0.5_b[0.5, 1.5]_lr0.001_lblock14_sepconv2_act',
        'preprocessing_function': xception_preprocessing,
        'rotation': 10,
        'shear': 0.25,
        'zoom': 0.5,
        'brightness': [0.5, 1.5],
        'lr': 0.001,
        'last_fixed_layer': 'block14_sepconv2_act',
        'model': get_xception_model((400, 300)),
        'base_path': "/home/ubuntu/hot-store/25e/xception",
        'image_size': (400, 300),
        'epochs': 10,
    },
]

data_path = '/home/ubuntu/hot-store/dermx/images'
iterations = 5

for model_settings in models:
    for iteration in range(iterations):
        try:
            validate_model(model_settings['base_path'], model_settings['model_name'] + f'_{iteration}',
                           model_settings['preprocessing_function'], data_path, model_settings['image_size'], remove_weights=False,
                           preds_path=Path(model_settings['base_path']) / f'dermx_{iteration}_preds.csv')
        except ResourceExhaustedError:
            print('Using batch size 32')
            validate_model(model_settings['base_path'], model_settings['model_name'] + f'_{iteration}',
                           model_settings['preprocessing_function'], data_path, model_settings['image_size'], remove_weights=False,
                           preds_path=Path(model_settings['base_path']) / f'dermx_{iteration}_preds.csv')

        K.clear_session()
        _ = gc.collect()
