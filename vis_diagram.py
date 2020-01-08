"""Visualize the actual diagram using the ckeckpoint."""
import os
import time
import glob
from collections import OrderedDict

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


def gen_diagram(model):
    # return digraph object
    x = torch.zeros(1, 3, 224, 224)
    digraph = make_dot(model(x), params=dict(model.named_parameters()))
    return digraph


def vis_diagram():
    # filter checkpoint
    ckpt_paths = glob.glob(os.path.join(cur_dir, "checkpoints", "*.pt"))
    ckpt_names = [p[p.rfind("/") + 1 :] for p in ckpt_paths]
    ckpt_names.insert(0, "-")
    selected_ckpt = st.selectbox("Please select the checkpoint", ckpt_names)

    # load selected
    model = None
    if selected_ckpt != "-":
        with st.spinner("loading {}...".format(selected_ckpt)):
            path = os.path.join(cur_dir, "checkpoints", selected_ckpt)
            model, missing_keys, unexpected_keys = load_model(path)
            if missing_keys:
                st.error("missing keys: {}".format(missing_keys))
                return -1

    # show button
    show = None
    if model:
        show = st.button("show")

    # show diagram
    if show:
        with st.spinner("generating diagram..."):
            digraph = gen_diagram(model)
            st.graphviz_chart(digraph)


def main():
    st.title("2D Unet Diagram")
    vis_diagram()


if __name__ == "__main__":
    main()
