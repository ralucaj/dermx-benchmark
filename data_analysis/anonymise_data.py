import os
from pathlib import Path

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

anonymise_annotations(dermx_path, labeller_dict)
anonymise_masks(dermx_masks_path, labeller_dict)
anonymise_masks_annotations(dermx_masks_annotation_path, labeller_dict)