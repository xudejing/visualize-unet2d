"""Visualize the conv filters as raw images."""
import os
import time
import glob

import torch
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


def gen_filter_image(model, layer_name):
    """ 
    Return:
        conv weights: the tensor [filter_num x input_channel x height_dim x width_dim]
        fig: only show first 64 filters, with full time dimension, average input channels to get gray image.
    """
    conv_weight_name = "model.{}.weight".format(layer_name)
    conv_weight = eval(conv_weight_name)
    # st.write(conv_weight_name, conv_weight.shape)
    conv_weight = conv_weight.permute([0, 2, 3, 1])  # (32,3,3,32)
    conv_weight = conv_weight.data.numpy()

    filter_num = conv_weight.shape[0]
    if filter_num > 64:
        filter_num = 64  # 64 is enough
    fig, axs = plt.subplots(1, filter_num)

    for i in range(filter_num):
        axs[i].axis("off")

    for i in range(filter_num):
        image = conv_weight[i]
        image = np.mean(image, axis=2)  # average the input channels
        image -= image.min()
        image /= image.max()
        axs[i].imshow(image, cmap="gray")

    return conv_weight, fig


def vis_filter_images():
    # filter checkpoint
    ckpt_paths = glob.glob(os.path.join(cur_dir, "checkpoints", "*.pt"))
    ckpt_names = [p[p.rfind("/") + 1 :] for p in ckpt_paths]
    selected_ckpts = st.multiselect(
        "Please select the checkpoints you want to explore", ckpt_names
    )

    # load selected
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

    # gather conv layers
    selected_layers = None
    if models:
        layer_names = []
        sample_ckpt = models[list(models.keys())[0]]
        for name, param in sample_ckpt.named_parameters():
            if "weight" in name and "norm" not in name and name != "conv.weight":
                name = name[: name.rfind(".")]
                layer_names.append(name)
        selected_layers = st.multiselect(
            "Please select the layers you want to compare", layer_names
        )

    # show buttons
    show = None
    if selected_layers:
        show = st.button("show")

    # show act maps
    if show:
        st.markdown(
            "_Only show first 64 filters, with full time dimension, average input channels to get gray image._"
        )
        for layer_name in selected_layers:
            st.subheader(layer_name)
            for name, model in models.items():
                st.text(name)
                with st.spinner("visualizing..."):
                    conv1_weight, fig = gen_filter_image(model, layer_name)
                st.pyplot(fig, bbox_inches="tight", pad_inches=0)


def main():
    st.title("2D Unet Filter Images")
    vis_filter_images()


if __name__ == "__main__":
    main()
