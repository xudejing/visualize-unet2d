"""Visualize activation maps for each channel."""
import os
import time
import glob
import random

import torch
from torchvision import transforms
from skimage.io import imread, imsave
from skimage.transform import resize
import torchvision.transforms.functional as TF
from PIL import Image
from torchviz import make_dot
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

try:
    from model import *
    from util import *
except ImportError:
    from .model import *
    from .util import *


def gen_act_lgg_mri(model, layer, patient, slice_id, channels):
    # register hook
    activation = {}

    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()

        return hook

    prefix = "model." + layer
    eval(prefix).register_forward_hook(get_activation(layer))

    # model inference
    patient_dirpath = os.path.join(cur_dir, "datasets", "lgg-mri-segmentation", patient)

    slice_filename = "{}_{}.tif".format(patient, slice_id)
    slice_path = os.path.join(patient_dirpath, slice_filename)
    input_image = Image.open(slice_path)
    m, s = np.mean(input_image, axis=(0, 1)), np.std(input_image, axis=(0, 1))
    input_tensor = torch.tensor((input_image - m) / s, dtype=torch.float)
    input_tensor = input_tensor.permute([2, 0, 1])

    input_batch = input_tensor.unsqueeze(0)
    input_batch = input_batch.to(device)
    model = model.to(device)
    model.eval()
    output = model(input_batch)

    # generate image
    if len(channels) < 3:
        fig_col = 3
    else:
        fig_col = len(channels)

    fig, axs = plt.subplots(2, fig_col, figsize=(fig_col, 2), sharex=True, sharey=True)
    remove_all_spines(axs)
    # pred mask
    pred_mask = output[0][0].detach().cpu().numpy()
    pred_mask = np.round(pred_mask)
    axs[0, 0].imshow(input_image)
    mask_filename = "{}_{}_mask.tif".format(patient, slice_id)
    mask_path = os.path.join(patient_dirpath, mask_filename)
    mask = imread(mask_path)
    axs[0, 1].imshow(mask, cmap="gray")
    axs[0, 2].imshow(pred_mask, cmap="gray")
    # activiation map
    for i, channel_id in enumerate(channels):
        channel_id = int(channel_id)
        A = activation[layer][0][channel_id].cpu().numpy()
        S = resize(A, (256, 256))
        axs[1, i].imshow(S, cmap="gray")
        axs[1, i].set_xlabel(channel_id)

    axs[1, 0].set_ylabel("act ({})".format(A.shape[0]))

    plt.xticks([])
    plt.yticks([])
    return fig


def vis_lgg_mri(models):
    # load patients
    patient_dirpath = glob.glob(
        os.path.join(cur_dir, "datasets", "lgg-mri-segmentation", "TCGA*")
    )
    patient_names = [p[p.rfind("/") + 1 :] for p in patient_dirpath]
    patient_names.insert(0, "-")
    selected_patient = st.selectbox("Please select the patient", patient_names)

    # show slice and its mask
    selected_slice_id = None
    if selected_patient != "-":
        fig, slice_num = gen_patient_lgg_mri(selected_patient)
        st.pyplot(fig, bbox_inches="tight", use_column_width=True)

        # select specific slice
        selected_slice_id = st.slider(
            "Please select the slice", min_value=1, max_value=slice_num
        )

    # select layer
    selected_layer = None
    if selected_slice_id:
        layer_names = []
        model = UNet(in_channels=3, out_channels=1, init_features=32)
        for name, param in model.named_children():
            layer_names.append(name)
        selected_layer = st.selectbox("Please select the layer", layer_names)

    # select channels
    selected_channels = None
    if selected_layer:
        # register hook
        activation = {}

        def get_activation(name):
            def hook(model, input, output):
                activation[name] = output.detach()

            return hook

        prefix = "model." + selected_layer
        eval(prefix).register_forward_hook(get_activation(selected_layer))
        model.eval()
        model(torch.zeros((1, 3, 256, 256)))
        num_act_maps = activation[selected_layer].shape[1]

        channels = [str(x) for x in list(range(num_act_maps))]
        if len(channels) <= 32:
            default_select = channels
        else:
            default_idxs = [int(x) for x in np.linspace(0, len(channels) - 1, 32)]
            default_select = [channels[idx] for idx in default_idxs]
        selected_channels = st.multiselect(
            "Please select the channels", channels, default_select
        )

    # show buttons
    show = None
    if selected_channels:
        show = st.button("show")

    # show act maps
    if show:
        st.write(selected_layer)
        for name, model in models.items():
            st.text(name)
            with st.spinner("generating activation maps..."):
                fig = gen_act_lgg_mri(
                    model,
                    selected_layer,
                    selected_patient,
                    selected_slice_id,
                    selected_channels,
                )
                st.pyplot(fig, bbox_inches="tight")


def vis_activation_maps():
    # filter checkpoint
    ckpt_paths = glob.glob(os.path.join(cur_dir, "checkpoints", "*.pt"))
    ckpt_names = [p[p.rfind("/") + 1 :] for p in ckpt_paths]
    ckpt_names.insert(0, "-")
    selected_ckpts = st.multiselect("Please select the checkpoints", ckpt_names)

    # load model
    models = {}
    if selected_ckpts:
        for name in selected_ckpts:
            with st.spinner("loading {}...".format(name)):
                path = os.path.join(cur_dir, "checkpoints", name)
                model, missing_keys, unexpected_keys = load_model(path)
                if missing_keys:
                    st.error("missing keys for {} \n\n {}".format(name, missing_keys))
                    return -1
                else:
                    models[name] = model

    # load dataset
    selected_dataset = None
    if models:
        # select dataset
        avaiable_dataset_paths = glob.glob(os.path.join(cur_dir, "datasets", "*/"))
        avaiable_dataset_names = [
            os.path.basename(path[:-1]) for path in avaiable_dataset_paths
        ]

        avaiable_dataset_names.insert(0, "-")
        selected_dataset = st.selectbox(
            "Please select the dataset", avaiable_dataset_names
        )

    # visualization
    if selected_dataset == "lgg-mri-segmentation":
        vis_lgg_mri(models)


def main():
    st.title("2D Unet Activation Maps (Channel)")
    vis_activation_maps()


if __name__ == "__main__":
    main()
