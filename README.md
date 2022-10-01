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

The program implements the classification and retrieval of 3D models through an approach based on graphs obtained from STEP files and the [MVCNN](https://github.com/jongchyisu/mvcnn_pytorch) approach based on multiple 2D views.

### Graph classification and retrieval

For the graph based approach, to convert a 3D STEP dataset into a Graph dataset, run the script:    
```
$ python step_2_graph.py
```    
It takes two arguments: `--path_stp` specifies the path of the input STEP dataset and `--path_graph` specifies the output path where the graph dataset will be saved.
Then for the classification task on the relised dataset run the script:   
```
$ python train_GCN.py
```
It takes 5 arguments: `--run_folder` indicates the run directory, `--learning_rate` sets the strating learning rate, `--batch_size` sets the batch size, `--num_epochs` sets the number of traing epochs, `--dropout` the dropout probability.    
Alternatively, we provide the `GCN_classification.ipynb` ipython notebook, that performs both the dataset conversion and graph classification task.   
A Graph Convolutional Neural Network model trained for the classification task in this way can then be used for the retrieval task by running the `GCN_retrieval.ipynb` script.

### Multi-views classification 

For the multi 2D views  based approach, to convert each 3D model into a 12 2D views,  run the script:
```
$ python step_2_multiview.py 
```
It takes two arguments: `--path_stp` specifies the path of the input STEP dataset and `--path_multiview` specifies the output path where the multi-views dataset will be saved.   
Then for the classification task run the script:
```
$ python train_mvcnn.py
```
It takes 10 arguments: `--num_models` indicates the number of models per class, `--lr` sets the strating learning rate, `--bs` sets the batch size, `--weight_decay` sets the weight decay ratio of the learning rate, `--num_epoch` sets the number of training epochs, `--no_pretraining` indicates if the base net will start pretrained or not, `--cnn_name` the net name, num_views the number of 2D views, `--train_path` specifies the path of the train data, `--test_path` specifies the path of the test data, `--val_path` specifies the path of the validation data.   
Alternatively, we provide a the `MultiViews_Classification.ipynb.ipynb` ipython notebook, that performs both the dataset conversion and multi-views classification task. 
Similarly to the graph-based approach, a model trained for classification task can then be used for the 3D retrieval task.

---

# Repository Requirements

This code was written in Pytorch 1.11. with CUDA Toolkit version 11.3 to enable GPU computations. We recommend setting up a virtual environment using [Miniconda](https://docs.conda.io/en/latest/miniconda.html). Python 3.8 is required for the PythonOCC library needed for the conversion from STEP to the multi-views data.

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
