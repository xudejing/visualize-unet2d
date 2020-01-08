"""Visualize the stats of conv filters."""
import os
import time
import glob

import torch
from torchviz import make_dot
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

try:
    from model import *
    from util import *
except ImportError:
    from .model import *
    from .util import *


def gen_filter_stats_matplotlib(model, layer_name):
    """ 
    Return:
        conv weights: the tensor [filter_num x time_dim x height_dim x width_dim x input_channel]
        fig: show distribution of the layer
    """
    conv_weight_name = "model.{}.weight".format(layer_name)
    conv_weight = eval(conv_weight_name)
    # print(conv_weight_name, conv_weight.shape)
    conv_weight = conv_weight.permute([0, 2, 3, 1])  # (32,3,3,32)
    conv_weight = conv_weight.data.numpy()

    filter_num = conv_weight.shape[0]
    fig, axs = plt.subplots(1, 1, figsize=(10, 1))

    filter_id = list(range(filter_num))
    filter_mean = np.mean(conv_weight, (1, 2, 3))
    axs.bar(filter_id, filter_mean)
    fig.tight_layout()
    return conv_weight, fig


def gen_filter_stats_plotly(model, layer_name):
    """ 
    Return:
        conv weights: the tensor [filter_num x time_dim x height_dim x width_dim x input_channel]
        fig: show distribution of the layer
    """
    conv_weight_name = "model.{}.weight".format(layer_name)
    conv_weight = eval(conv_weight_name)
    # print(conv_weight_name, conv_weight.shape)
    conv_weight = conv_weight.permute([0, 2, 3, 1])  # (32,3,3,32)
    conv_weight = conv_weight.data.numpy()

    filter_num = conv_weight.shape[0]

    filter_id = list(range(filter_num))
    filter_mean = np.mean(conv_weight, (1, 2, 3))

    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(go.Bar(x=filter_id, y=filter_mean), row=1, col=1)

    fig.update_layout(
        template="plotly_white", margin=go.layout.Margin(l=0, r=0, b=0, t=0, pad=0)
    )

    return conv_weight, fig


def vis_filter_stats():
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
        for layer_name in selected_layers:
            st.subheader(layer_name)
            for name, model in models.items():
                st.text(name)
                with st.spinner("visualizing..."):
                    conv1_weight, fig = gen_filter_stats_plotly(model, layer_name)
                    # conv1_weight, fig = gen_filter_stats_matplotlib(
                    #     model, layer_name)
                st.plotly_chart(fig, height=200)
                # st.pyplot(fig, bbox_inches="tight", pad_inches=0)


def main():
    st.title("2D Unet Filter Stats")
    vis_filter_stats()


if __name__ == "__main__":
    main()
