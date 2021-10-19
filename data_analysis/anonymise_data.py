import os
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image


# TODO: replace a, b, ... with actual names before running this file
labeller_dict = {
    'a': 'labeller1',
    'b': 'labeller2',
    'c': 'labeller3',
    'd': 'labeller4',
    'e': 'labeller5',
    'f': 'labeller6',
    'g': 'labeller7',
    'h': 'labeller8',
}

dermx_path = '/mnt/c/Users/raluca/Downloads/DermX-v2'
dermx_masks_path = '/mnt/c/Users/raluca/Downloads/DermX-v2-masks/masks'
dermx_masks_annotation_path = '/mnt/c/Users/raluca/Downloads/DermX-v2-masks/instance_mask_annotations.csv'


def anonymise_annotations(folder, rename_dict):
    """
    Replace labeller names in annotation files saved in `folder` with an equivalent from `rename_dict`
    :param folder: location of annotation files
    :param rename_dict: dict from labeller name to anonymised name
    :return:
    """
    filenames = os.listdir(folder)
    for filename in filenames:
        try:
            labeller, image = filename.split('_')
        except ValueError:
            print(filename)
            os.remove(Path(folder) / filename)
            continue

        # Replace labeller names in the JSON files
        annotation_path = Path(folder) / filename
        annotation = annotation_path.read_text().replace(labeller, rename_dict[labeller])
        annotation_path.write_text(annotation)
        # Rename files according to the rename dict
        os.rename(Path(folder) / filename, Path(folder) / (rename_dict[labeller] + '_' + image))


def anonymise_masks(folder, rename_dict):
    """
    Replace labeller names in mask files saved in `folder` with an equivalent from `rename_dict`
    :param folder: location of mask files
    :param rename_dict: dict from labeller name to anonymised name
    :return:
    """
    filenames = os.listdir(folder)
    for filename in filenames:
        try:
            labeller, image, characteristic = filename.split('_')
        except ValueError:
            print(filename)
            os.remove(Path(folder) / filename)
            continue

        # Rename files according to the rename dict
        os.rename(Path(folder) / filename, Path(folder) / (rename_dict[labeller] + '_' + image + '_' + characteristic))


def anonymise_masks_annotations(filename, rename_dict):
    """
    Replace labeller names in the mask annotation file `filename` with an equivalent from `rename_dict`
    :param filename: location of the mask annotation csv file
    :param rename_dict:  dict from labeller name to anonymised name
    :return:
    """
    # Replace labeller names in the JSON files
    annotation_path = Path(filename)
    annotation = annotation_path.read_text()
    for labeller, anon_labeller in rename_dict.items():
        annotation = annotation.replace(labeller, anon_labeller)
    annotation_path.write_text(annotation)


def get_masks_info(masks_metadata_path):
    masks_info_df = pd.read_csv(masks_metadata_path)
    masks_info_df[['labeller', 'image', 'characteristic', 'empty']] = masks_info_df.mask_id.str.split('_', expand=True)
    masks_info_df = masks_info_df.drop(columns=['empty'])
    return masks_info_df


def clean_up_masks(masks_path, metadata_path):
    """
    Clean up mask data. First, rename masks as labeller-id_image-id_characteristic-name. Then, merge masks of the same
    characteristic into a single mask. E.g. if labeller1 outlined two plaques in image1, the two masks will become a
    single mask named labeller1_image1_plaque.png.
    :param metadata_path: path to the masks metadata csv
    :param masks_path: path to the masks folder
    :return:
    """
    masks_info_df = get_masks_info(metadata_path)

    for row in masks_info_df.iterrows():
        old_name = Path(masks_path) / (row[1]['mask_id'] + '.png')
        new_name = Path(masks_path) / (row[1]['image_id'] + '_' + row[1]['class_name'] + '.png')
        if not os.path.exists(old_name):
            # Ignore duplicate images
            continue
        if os.path.exists(new_name):
            image_1 = np.array(Image.open(new_name))
            image_2 = np.array(Image.open(old_name))
            combined = np.logical_or(image_1, image_2).astype('uint8') * 255
            Image.fromarray(combined).save(new_name)
        else:
            os.rename(old_name, new_name)


anonymise_annotations(dermx_path, labeller_dict)
anonymise_masks(dermx_masks_path, labeller_dict)
anonymise_masks_annotations(dermx_masks_annotation_path, labeller_dict)
clean_up_masks(dermx_masks_path, dermx_masks_annotation_path)
