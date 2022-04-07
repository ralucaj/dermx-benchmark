from utils import train_model, unfreeze_layers, get_resnet_model, get_efficientnet_model, validate_model
from tensorflow.keras.applications.resnet import preprocess_input as resnet_preprocessing
from tensorflow.errors import ResourceExhaustedError


models = [
    {
        'model_base_name': 'resnet50',
        'preprocessing_function': resnet_preprocessing,
        'last_fixed_layers':['conv5_block3_out', 'conv5_block2_add'],
        'model': get_resnet_model(),
        'base_path': "/home/ubuntu/store/resnet-hpo"
    },
    {
        'model_base_name': 'efficientnetb0',
        'preprocessing_function': None,
        'last_fixed_layers': ['top_conv', 'block6d_add'],
        'model': get_efficientnet_model(),
        'base_path': "/home/ubuntu/store/efficientnet-hpo"
    },
]
data_path = '/home/ubuntu/store/barankin-neurips/hpo'
rotation_ranges = [10, 20]
shear_ranges = [0, 0.25, 0.5]
zoom_ranges = [0.25, 0.5]
brightness_ranges = [[0.25, 0.5], [0.5, 1], [0.25, 1]]
learning_rates = [0.01, 0.001, 0.0001]

for model_settings in models:
    for rotation in rotation_ranges:
        for shear in shear_ranges:
            for zoom in zoom_ranges:
                for brightness in brightness_ranges:
                    for lr in learning_rates:
                        for last_fixed_layer in model_settings['last_fixed_layers']:
                            model = unfreeze_layers(model_settings['model'], last_fixed_layer)
                            try:
                                model_name = train_model(model, model_settings['model_base_name'], rotation, shear, zoom, brightness, lr, last_fixed_layer, 64, model_settings['preprocessing_function'], model_settings['base_path'], data_path)
                            except ResourceExhaustedError:
                                print('Using batch size 32')
                                model_name = train_model(model, model_settings['model_base_name'], rotation, shear, zoom, brightness, lr, last_fixed_layer, 32, model_settings['preprocessing_function'], model_settings['base_path'], data_path)
                            validate_model(model_settings['base_path'], model_name, model_settings['preprocessing_function'], data_path)
