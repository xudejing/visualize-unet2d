# Visualize 2D Unet
This is a **template project** that demonstrates the code structure. The template can be used as follows:
* one project of the [visualize-neural-networks](https://github.com/xudejing/visualize-neural-networks);
* standalone;
* display any single visualization (easy for debug and collaboration);

The model we explore here is from [PyTorch Hub](https://pytorch.org/hub/mateuszbuda_brain-segmentation-pytorch_unet/).

## Installation 
1. Clone the repository to your local machine

    ```
    $ git clone https://github.com/xudejing/visualize-unet2d.git
    ```

2. Install [PyTorch](https://pytorch.org) by following the official instructions (conda prefered).

3. Install other dependency packages.
   ```
   pip install -r requirements.txt
   ```
    
4. Download the [checkpoints](https://mega.nz/#F!JyZABQDZ!gah8I_xCrj7aTKubaKJ7zw) and put them under *checkpoints*. You can add more checkpoints and explore them together.

5. Download the [LGG Segmentation Dataset](https://www.kaggle.com/mateuszbuda/lgg-mri-segmentation) and extract them under *datasets*.

6. Run the app
    ```
    $ streamlit run app.py
    ```


## Structure
Each visualization is organized in a specific module named with prefix *vis_*, for example here we have 5 visualizations which are:
* Diagram (*vis_dirgram.py*)
* Filter Images (*vis_filter_image.py*)
* Filter Stats (*vis_filter_stats.py*)
* Activation Map (*vis_act_map.py*)
* Activation Map (Channel) (*vis_act_map_channel.py*)

If you create new visualizations (related to 2D Unet), you also need to register them in *app.py*.

Inside each module, it is better to split the computation and visualization code as separate functions, such as *gen_sth()* and *vis_sth()*. This can lighten the load if you also want to use the generated image in other situations.

## Single visualization
You can even run a single visualization by change *app.py* in the running command to other modules start with the prefix *vis_*.
