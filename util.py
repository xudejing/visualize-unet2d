"""Utils"""
import os
import glob

import torch
import matplotlib.pyplot as plt
from skimage.io import imread

try:
    from model import UNet
except ImportError:
    from .model import UNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cur_dir = os.path.dirname(os.path.abspath(__file__))


def load_model(path):
    model = UNet(in_channels=3, out_channels=1, init_features=32)
    state_dict = torch.load(path, map_location=device)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict)
    return model, missing_keys, unexpected_keys


def gen_patient_lgg_mri(patient):
    patient_dirpath = os.path.join(cur_dir, "datasets", "lgg-mri-segmentation", patient)
    filenames = glob.glob(os.path.join(patient_dirpath, "*.tif"))
    slice_num = len(filenames) // 2
    fig, axs = plt.subplots(
        2, slice_num, figsize=(slice_num, 2), sharex=True, sharey=True
    )
    for slice_id in range(1, slice_num + 1):
        # mri
        mri_filename = "{}_{}.tif".format(patient, slice_id)
        mri_path = os.path.join(patient_dirpath, mri_filename)
        mri = imread(mri_path)
        axs[0, slice_id - 1].imshow(mri)
        # gt mask
        mask_filename = "{}_{}_mask.tif".format(patient, slice_id)
        mask_path = os.path.join(patient_dirpath, mask_filename)
        mask = imread(mask_path)
        axs[1, slice_id - 1].imshow(mask, cmap="gray")
        axs[1, slice_id - 1].set_xlabel(slice_id)

    axs[0, 0].set_ylabel("mri")
    axs[1, 0].set_ylabel("gt")

    plt.xticks([])
    plt.yticks([])
    return fig, slice_num


def remove_all_spines(axs):
    for ax in axs.reshape(-1):
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
