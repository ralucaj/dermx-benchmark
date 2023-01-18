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

models = [
    {
       'model_base_name': 'resnet50',
       'preprocessing_function': resnet_preprocessing,
       'last_fixed_layers': ['conv5_block3_out', 'conv5_block2_add'],
       'model': get_resnet_model((400, 300)),
       'base_path': "/home/ubuntu/hot-store/resnet-hpo",
       'image_size': (400, 300)
    },
    {
       'model_base_name': 'efficientnetb0',
       'preprocessing_function': efficientnet_preprocessing,
       'last_fixed_layers': ['top_conv', 'block6d_add'],
       'model': get_efficientnet_model((400, 300)),
       'base_path': "/home/ubuntu/hot-store/efficientnet-hpo",
       'image_size': (400, 300)
    },
    {
       'model_base_name': 'inceptionresnetv2',
       'preprocessing_function': inceptionresnet_preprocessing,
       'last_fixed_layers': ['conv_7b_ac', 'block8_9_ac'],
       'model': get_inceptionresnet_model((400, 300)),
       'base_path': "/home/ubuntu/hot-store/inceptionresnet-hpo",
       'image_size': (400, 300)
    },
    {
       'model_base_name': 'inception',
       'preprocessing_function': inception_preprocessing,
       'last_fixed_layers': ["mixed10", "activation_288"],
       'model': get_inception_model((400, 300)),
       'base_path': "/home/ubuntu/hot-store/inception-hpo",
       'image_size': (400, 300)
    },
    {
       'model_base_name': 'mobilenetv1',
       'preprocessing_function': mobilenet_preprocessing,
       'last_fixed_layers': ["conv_pw_13_relu", "conv_pw_12_relu"],
       'model': get_mobilenetv1_model((400, 300)),
       'base_path': "/home/ubuntu/hot-store/mobilenetv1-hpo",
       'image_size': (400, 300)
    },
    {
       'model_base_name': 'mobilenetv2',
       'preprocessing_function': mobilenetv2_preprocessing,
       'last_fixed_layers': ["out_relu", "block_15_add"],
       'model': get_mobilenetv2_model((400, 300)),
       'base_path': "/home/ubuntu/hot-store/mobilenetv2-hpo",
       'image_size': (400, 300)
    },
    {
       'model_base_name': 'nasnetmobile',
       'preprocessing_function': nasnet_preprocessing,
       'last_fixed_layers': ["activation_187", "normal_concat_11"],
       'model': get_nasnetmobile_model((224, 224)),
       'base_path': "/home/ubuntu/hot-store/nasnetmobile-hpo",
       'image_size': (224, 224)
    },
    {
       'model_base_name': 'resnetv2',
       'preprocessing_function': resnetv2_preprocessing,
       'last_fixed_layers': ["post_relu", "conv5_block2_out"],
       'model': get_resnetv2_model((400, 300)),
       'base_path': "/home/ubuntu/hot-store/resnetv2-hpo",
       'image_size': (400, 300)
    },
    {
        'model_base_name': 'vgg',
        'preprocessing_function': vgg_preprocessing,
        'last_fixed_layers': ["block5_pool", "block4_pool"],
        'model': get_vgg_model((400, 300)),
        'base_path': "/home/ubuntu/hot-store/vgg-small-hpo",
        'image_size': (400, 300)
    },
    {
        'model_base_name': 'xception',
        'preprocessing_function': xception_preprocessing,
        'last_fixed_layers': ["block14_sepconv2_act", "add_11"],
        'model': get_xception_model((400, 300)),
        'base_path': "/home/ubuntu/hot-store/xception-hpo",
        'image_size': (400, 300)
    },
]
data_path = '/home/ubuntu/store/barankin-neurips/hpo'
rotation_ranges = [10, 20]
shear_ranges = [0.25, 0.5]
zoom_ranges = [0.25, 0.5]
brightness_ranges = [[0.75, 1.25], [0.5, 1.5]]
learning_rates = [0.01, 0.001, 0.0001]

for model_settings in models:
    if not os.path.isdir(model_settings['base_path']):
        os.makedirs(model_settings['base_path'])
    for rotation in rotation_ranges:
        for shear in shear_ranges:
            for zoom in zoom_ranges:
                for brightness in brightness_ranges:
                    for lr in learning_rates:
                        for last_fixed_layer in model_settings['last_fixed_layers']:
                            model = unfreeze_layers(model_settings['model'], last_fixed_layer)
                            try:
                                model_name = train_model(model, model_settings['model_base_name'], rotation, shear,
                                                         zoom, brightness, lr, last_fixed_layer, 64,
                                                         model_settings['preprocessing_function'],
                                                         model_settings['base_path'], data_path, model_settings['image_size'])
                                validate_model(model_settings['base_path'], model_name,
                                               model_settings['preprocessing_function'], data_path, model_settings['image_size'])
                            except ResourceExhaustedError:
                                print('Using batch size 32')
                                model_name = train_model(model, model_settings['model_base_name'], rotation, shear,
                                                         zoom, brightness, lr, last_fixed_layer, 32,
                                                         model_settings['preprocessing_function'],
                                                         model_settings['base_path'], data_path, model_settings['image_size'])
                                validate_model(model_settings['base_path'], model_name,
                                               model_settings['preprocessing_function'], data_path, model_settings['image_size'])
                                
                            K.clear_session()
                            _ = gc.collect()
