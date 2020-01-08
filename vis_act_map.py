"""Visualize activation maps by averaging channels."""
import os
import time
import glob

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


def gen_act_lgg_mri(model, layers, patient):
    # register hook
    activation = {}

    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()

        return hook

    for layer_name in layers:
        prefix = "model." + layer_name
        eval(prefix).register_forward_hook(get_activation(layer_name))

    # model inference
    patient_dirpath = os.path.join(cur_dir, "datasets", "lgg-mri-segmentation", patient)
    filenames = glob.glob(os.path.join(patient_dirpath, "*.tif"))
    slice_num = len(filenames) // 2
    input_batch = []

    for slice_id in range(1, slice_num + 1):
        mri_filename = "{}_{}.tif".format(patient, slice_id)
        mri_path = os.path.join(patient_dirpath, mri_filename)
        input_image = Image.open(mri_path)
        m, s = np.mean(input_image, axis=(0, 1)), np.std(input_image, axis=(0, 1))
        input_tensor = torch.tensor((input_image - m) / s, dtype=torch.float)
        input_tensor = input_tensor.permute([2, 0, 1])
        input_batch.append(input_tensor)

    input_batch = torch.stack(input_batch)
    input_batch = input_batch.to(device)
    model = model.to(device)
    model.eval()
    output = model(input_batch)

    # generate image
    fig, axs = plt.subplots(
        1 + len(layers),
        slice_num,
        figsize=(slice_num, 1 + len(layers)),
        sharex=True,
        sharey=True,
    )
    remove_all_spines(axs)
    for slice_id in range(slice_num):
        # pred mask
        pred_mask = output[slice_id][0].detach().cpu().numpy()
        pred_mask = np.round(pred_mask)
        axs[0, slice_id].imshow(pred_mask, cmap="gray")

        # activiation map
        act_row_id = 1
        for layer_name in layers:
            # A = activation[layer_name][slice_id][0].cpu().numpy() # just the first channel
            A = activation[layer_name][slice_id].cpu().numpy()
            A = np.average(A, axis=0)  # average
            S = resize(A, (256, 256))
            axs[act_row_id, slice_id].imshow(S, cmap="gray")
            act_row_id += 1
        # add slice id
        axs[-1, slice_id].set_xlabel(slice_id + 1)

    axs[0, 0].set_ylabel("pred")
    act_row_id = 1
    i = 0
    for layer_name in layers:
        axs[act_row_id + i, 0].set_ylabel(i)
        i += 1

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

    # show mri and its mask
    selected_layers = None
    if selected_patient != "-":
        fig, _ = gen_patient_lgg_mri(selected_patient)
        st.pyplot(fig, bbox_inches="tight")

        # select layers
        layer_names = []
        model = UNet(in_channels=3, out_channels=1, init_features=32)
        for name, param in model.named_children():
            layer_names.append(name)
        selected_layers = st.multiselect("Please select the layers", layer_names)

    # show buttons
    show = None
    if selected_layers:
        show = st.button("show")

    # show act maps
    if show:
        st.write(selected_layers)
        for name, model in models.items():
            st.text(name)
            with st.spinner("generating activation maps..."):
                fig = gen_act_lgg_mri(model, selected_layers, selected_patient)
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
    st.title("2D Unet Activation Maps")
    vis_activation_maps()


if __name__ == "__main__":
    main()
