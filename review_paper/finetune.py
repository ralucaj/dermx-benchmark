import tensorflow as tf
from utils import train_model, unfreeze_layers, get_resnet_model, get_efficientnet_model, validate_model, \
    get_xception_model, get_vgg_model, get_resnetv2_model, get_nasnetmobile_model, get_mobilenetv2_model, \
    get_mobilenetv1_model, get_inception_model, get_inceptionresnet_model, finetune_model, test_model
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
from tensorflow.keras.models import load_model
from pathlib import Path

import os
import gc
import glob
from tensorflow.keras import backend as K


models = [
#     {
#         'model_base_name': 'resnet50',
#         'preprocessing_function': resnet_preprocessing,
#         'rotation': 20,
#         'shear': 0.5,
#         'zoom': 0.5,
#         'brightness': [0.5, 1.5],
#         'lr': 0.0001,
#         'last_fixed_layer': 'conv5_block3_out',
#         'model': get_resnet_model((400, 300)),
#         'base_path': "/home/ubuntu/hot-store/final/resnet",
#         'image_size': (400, 300),
#         'epochs': 50,
#     },
#     {
#         'model_base_name': 'efficientnetb0',
#         'preprocessing_function': efficientnet_preprocessing,
#         'rotation': 20,
#         'shear': 0.25,
#         'zoom': 0.5,
#         'brightness': [0.5, 1.5],
#         'lr': 0.0001,
#         'last_fixed_layer': 'block6d_add',
#         'model': get_efficientnet_model((400, 300)),
#         'base_path': "/home/ubuntu/hot-store/final/efficientnet",
#         'image_size': (400, 300),
#         'epochs': 50,
#     },
#     {
#         'model_base_name': 'inceptionresnetv2',
#         'preprocessing_function': inceptionresnet_preprocessing,
#         'rotation': 20,
#         'shear': 0.25,
#         'zoom': 0.5,
#         'brightness': [0.75, 1.25],
#         'lr': 0.0001,
#         'last_fixed_layer': 'block8_9_ac',
#         'model': get_inceptionresnet_model((400, 300)),
#         'base_path': "/home/ubuntu/hot-store/final/inceptionresnet",
#         'image_size': (400, 300),
#         'epochs': 50,
#     },
#     {
#         'model_base_name': 'inception',
#         'preprocessing_function': inception_preprocessing,
#         'rotation': 20,
#         'shear': 0.5,
#         'zoom': 0.5,
#         'brightness': [0.5, 1.5],
#         'lr': 0.001,
#         'last_fixed_layer': 'activation_288',
#         'model': get_inception_model((400, 300)),
#         'base_path': "/home/ubuntu/hot-store/final/inception",
#         'image_size': (400, 300),
#         'epochs': 50,
#     },
#     {
#         'model_base_name': 'mobilenetv1',
#         'preprocessing_function': mobilenet_preprocessing,
#         'rotation': 10,
#         'shear': 0.5,
#         'zoom': 0.5,
#         'brightness': [0.5, 1.5],
#         'lr': 0.0001,
#         'last_fixed_layer': 'conv_pw_12_relu',
#         'model': get_mobilenetv1_model((400, 300)),
#         'base_path': "/home/ubuntu/hot-store/final/mobilenetv1",
#         'image_size': (400, 300),
#         'epochs': 50,
#     },
#     {
#         'model_base_name': 'mobilenetv2',
#         'preprocessing_function': mobilenetv2_preprocessing,
#         'rotation': 10,
#         'shear': 0.25,
#         'zoom': 0.5,
#         'brightness': [0.5, 1.5],
#         'lr': 0.0001,
#         'last_fixed_layer': 'block_15_add',
#         'model': get_mobilenetv2_model((400, 300)),
#         'base_path': "/home/ubuntu/hot-store/final/mobilenetv2",
#         'image_size': (400, 300),
#         'epochs': 50,
#     },
#     {
#         'model_base_name': 'nasnetmobile',
#         'preprocessing_function': nasnet_preprocessing,
#         'rotation': 20,
#         'shear': 0.25,
#         'zoom': 0.5,
#         'brightness': [0.5, 1.0],
#         'lr': 0.0001,
#         'last_fixed_layer': 'normal_concat_11',
#         'model': get_nasnetmobile_model((224, 224)),
#         'base_path': "/home/ubuntu/hot-store/final/nasnetmobile",
#         'image_size': (224, 224),
#         'epochs': 50,
#     },
#     {
#         'model_base_name': 'resnetv2',
#         'preprocessing_function': resnetv2_preprocessing,
#         'rotation': 20,
#         'shear': 0.25,
#         'zoom': 0.25,
#         'brightness': [0.5, 1.0],
#         'lr': 0.001,
#         'last_fixed_layer': 'post_relu',
#         'model': get_resnetv2_model((400, 300)),
#         'base_path': "/home/ubuntu/hot-store/final/resnetv2",
#         'image_size': (400, 300),
#         'epochs': 50,
#     },
#     {
#         'model_base_name': 'vgg',
#         'preprocessing_function': vgg_preprocessing,
#         'rotation': 10,
#         'shear': 0.0,
#         'zoom': 0.25,
#         'brightness': [0.5, 1.0],
#         'lr': 0.01,
#         'last_fixed_layer': 'block5_pool',
#         'model': get_vgg_model((400, 300)),
#         'base_path': "/home/ubuntu/hot-store/final/vgg",
#         'image_size': (400, 300),
#         'epochs': 50,
#     },
    {
        'model_base_name': 'xception',
        'preprocessing_function': xception_preprocessing,
        'rotation': 10,
        'shear': 0.25,
        'zoom': 0.5,
        'brightness': [0.5, 1.5],
        'lr': 0.001,
        'last_fixed_layer': 'block14_sepconv2_act',
        'model': get_xception_model((400, 300)),
        'base_path': "/home/ubuntu/hot-store/final/xception",
        'image_size': (400, 300),
        'epochs': 50,
    },
]

data_path = '/home/ubuntu/hot-store/dermx_finetuning/split1'

for model_settings in models:
    model_paths = glob.glob(model_settings["base_path"] + '/*[0-4].h5')
    for model_path in model_paths:
        model = load_model(model_path)
        model_base_name = Path(model_path).stem
        if not os.path.isdir(model_settings['base_path']):
            os.makedirs(model_settings['base_path'])
        model = unfreeze_layers(model_settings['model'], model_settings['last_fixed_layer'])
        try:
            model_name = finetune_model(model, model_base_name, model_settings['rotation'], model_settings['shear'],
                                     model_settings['zoom'], model_settings['brightness'], model_settings['lr'],
                                     model_settings['last_fixed_layer'], 64, model_settings['preprocessing_function'], 
                                     model_settings['base_path'], data_path, model_settings['image_size'], name_suffix='_finetuned', 
                                     epochs=model_settings['epochs'])

            test_model(model_settings['base_path'], model_name,
                           model_settings['preprocessing_function'], data_path, model_settings['image_size'], valid_path='test')
        except ResourceExhaustedError:
            print('Using batch size 32')
            model_name = finetune_model(model, model_base_name, model_settings['rotation'], model_settings['shear'],
                                     model_settings['zoom'], model_settings['brightness'], model_settings['lr'],
                                     model_settings['last_fixed_layer'], 32,
                                     model_settings['preprocessing_function'],
                                     model_settings['base_path'], data_path, model_settings['image_size'], name_suffix=f'_finetuned',
                                     epochs=model_settings['epochs'])
            test_model(model_settings['base_path'], model_name,
                           model_settings['preprocessing_function'], data_path, model_settings['image_size'], valid_path='test')

        K.clear_session()
        _ = gc.collect()
