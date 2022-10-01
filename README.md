# CAD 3D Model classification by Graph Neural Networks: A new approach based on STEP format
## Authors: Lorenzo Mandelli, Stefano Berretti
#### Università degli Studi di Firenze

![](https://img.shields.io/github/contributors/divanoLetto/3D_STEP_Classification?color=light%20green) ![](https://img.shields.io/github/repo-size/divanoLetto/3D_STEP_Classification)

## Abstract
*In this paper, we introduce a new approach for retrieval and classification of 3D models that directly performs in the CAD
format without any format conversion to other representations like point clouds of meshes, thus avoiding any loss of informa-
tion. Among the various CAD formats, we consider the widely used STEP extension, which represents a standard for product
manufacturing information. This particular format represents a 3D model as a set of primitive elements such as surfaces and
vertices linked together. In our approach, we exploit the linked structure of STEP files to create a graph in which the nodes are
the primitive elements and the arcs are the connections between them. We then use Graph Neural Networks (GNNs) to solve the
problem of model classification. Finally, we created two datasets of 3D models in native CAD format, respectively, by collecting
data from the Traceparts model library and from the Configurators software modeling company. We used these datasets to test
and compare our approach with respect to state-of-the-art methods that consider other 3D formats*

Details about the implementation and the obtained results can be found in the `docs` folder.

---

## Installation

1. Create Conda virtual environment:

    ```
    conda create --name 3D_STEP_Classification python=3.8
    conda activate 3D_STEP_Classification
    ```
    
2. Clone this repository:
    ```
    git clone https://github.com/divanoLetto/3D_STEP_Classification
    ```
3. Install CUDA Toolkit version 11.3 from the [official site](https://developer.nvidia.com/cuda-11.3.0-download-archive).

4.  Install the following requirements:
    ```
    conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
    conda install pyg -c pyg
    conda install -c conda-forge tensorboardx
    conda install -c anaconda scikit-learn
    conda install -c conda-forge matplotlib
    conda install -c anaconda scikit-image
    conda install -c conda-forge pythonocc-core
    ```

5. Finally, make sure to obtain the [Traceparts STEP dataset](https://drive.google.com/drive/folders/1jV1B5Y8XmGY-XhjildX2BdYTEFtLK5XQ?usp=sharing), extract the STEP models and save them in the `/Datasets/` folder.

## Usage

In order to run the program for the first time using the default settings (those that yield the best results), simply run `python3 main.py`.
 
Two optional arguments, `--settings` and `--mode`, allow to choose the settings file to be used and the mode (`train` or `test`), e.g.: `python3 main.py --settings my_fav_settings_file --mode train`.

Please note that, unlike the original project, `data_generation.py` does not need to be run anymore if `split_dataset = True` in the chosen settings file. The `settings` folder has been specially created in order to contain any number of settings files, thus allowing for an easier setup of the model.


The following is the original abstract of the project.

---


![Neural3DMM architecture](images/architecture_figure1.png "Neural3DMM architecture")


# Repository Requirements

This code was written in Pytorch 1.1. We use tensorboardX for the visualisation of the training metrics. We recommend setting up a virtual environment using [Miniconda](https://docs.conda.io/en/latest/miniconda.html). To install Pytorch in a conda environment, simply run 

```
$ conda install pytorch torchvision -c pytorch
```

Then the rest of the requirements can be installed with 

```
$ pip install -r requirements.txt
```

### Mesh Decimation
For the mesh decimation code we use a function from the [COMA repository](https://github.com/anuragranj/coma) (the files **mesh_sampling.py** and **shape_data.py** - previously **facemesh.py** - were taken from the COMA repo and adapted to our needs). In order to decimate your template mesh, you will need the [MPI-Mesh](https://github.com/MPI-IS/mesh) package (a mesh library similar to Trimesh or Open3D).  This package requires Python 2. However once you have cached the generated downsampling and upsampling matrices, it is possible to run the rest of the code with Python 3 as well, if necessary.


# Data Organization

The following is the organization of the dataset directories expected by the code:

* data **root_dir**/
  * **dataset** name/ (eg DFAUST)
    * template
      * template.obj (all of the spiraling and downsampling code is run on the template only once)
      * downsample_method/
        * downsampling_matrices.pkl (created by the code the first time you run it)
    * preprocessed/
      * train.npy (number_meshes, number_vertices, 3) (no Faces because they all share topology)
      * test.npy 
      * points_train/ (created by data_generation.py)
      * points_val/ (created by data_generation.py)
      * points_test/ (created by data_generation.py)
      * paths_train.npy (created by data_generation.py)
      * paths_val.npy (created by data_generation.py)
      * paths_test.npy (created by data_generation.py)

# Usage

#### Data preprocessing 

In order to use a pytorch dataloader for training and testing, we split the data into seperate files by:

```
$ python data_generation.py --root_dir=/path/to/data_root_dir --dataset=DFAUST --num_valid=100
```

#### Training and Testing

For training and testing of the mesh autoencoder, we provide an ipython notebook, which you can run with 

```
$ jupyter notebook neural3dmm.ipynb
```

The first time you run the code, it will check if the downsampling matrices are cached (calculating the downsampling and upsampling matrices takes a few minutes), and then the spirals will be calculated on the template (**spiral_utils.py** file).

In the 2nd cell of the notebook one can specify their directories, hyperparameters (sizes of the filters, optimizer) etc. All this information is stored in a dictionary named _args_ that is used throughout the rest of the code. In order to run the notebook in train or test mode, simply set:

```
args['mode'] = 'train' or 'test'
```

#### Some important notes:
* The code has compatibility with both _mpi-mesh_ and _trimesh_ packages (it can be chosen by setting the _meshpackage_ variable in the first cell of the notebook).
* The reference points parameter needs exactly one vertex index per disconnected component of the mesh. So for DFAUST you only need one, but for COMA which has the eyes as diconnected components, you need a reference point on the head as well as one on each eye.
* **spiral_utils.py**: In order to get the spiral ordering for each neighborhood, the spiraling code works by walking along the triangulation exploiting the fact that the triangles are all listed in a consistent way (either clockwise or counter-clockwise). These are saved as lists (their length depends on the number of hops and number of neighbors), which are then truncated or padded with -1 (index to a dummy vertex) to match all the spiral lengths to a predefined value L (in our case L = mean spiral length + 2 standard deviations of the spiral lengths). These are used by the _SpiralConv_ function in **models.py**, which is the main module of our proposed method.

# Cite

Please consider citing our work if you find it useful:

```
@InProceedings{bouritsas2019neural,
    author = {Bouritsas, Giorgos and Bokhnyak, Sergiy and Ploumpis, Stylianos and Bronstein, Michael and Zafeiriou, Stefanos},
    title = {Neural 3D Morphable Models: Spiral Convolutional Networks for 3D Shape Representation Learning and Generation},
    booktitle   = {The IEEE International Conference on Computer Vision (ICCV)},
    year        = {2019}
}
```
