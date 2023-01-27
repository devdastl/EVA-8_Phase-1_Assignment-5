# EVA-8_Phase-1_Assignment-5
This is the assignment of 5th session in phase-1 of EVA-8 from TSAI.

## Introduction

### Objective
Objective of this assignment is to build three CNN based network which will take [MNIST Handwritten Dataset](<http://yann.lecun.com/exdb/mnist/>) and should have following properties.
1. This three networks should be following:
    - First network using Batch Normalization and L1 regularization.
    - Second network using Group Normalization
    - Third network using Layer Normalization.
2. We should have single model.py file which will contain model architecture defination as well as way to select from above mentioned normalization.
3. There should be a single notebook which will call this components and each model will be trained for 20epoch.
4. Graphs generated by the model as well as 10 misclassified images from each model.

### Repository setup
Since all the essential modules are written in .py files which are then getting called in main notebook, it is necessary to understand the structure of the repository.
Below is a quick look on how the repository is setup:
<br>
```
EVA-8_Phase-1_Assignment-5/
  |
  ├── EVA_Assignment-5.ipynb    <- Main colab notebook which will call other modules and perform training
  |
  ├── README.md                           <- The top-level README for developers using this project.
  |
  ├── LICENSE                             <- Standered apache 2.0 based license
  |
  ├── component/
  │   ├── data.py             <- Python file to download, process and create dataset for training.
  │   ├── model.py            <- Python file where model arcitecture is defined and can be changed if required.
  │   ├── training.py         <- Python file where training code is defined. Forward and backward pass will be done by this.
  │   ├── test.py             <- File to perform evaluation while training the model. It on performs forward pass with no gradient calc.
  │   ├── transform.py        <- We define test and train transformations/augmentation in this file.
  │   └── plot_util.py        <- Contains utility function to plot graphs and images.
  │
  ├── data/                   <- Directory to store data downloaded via torchvision
  │   ├── MNIST               <- Example when MNIST data will be downloaded
  │   ├── ImageNet            <- Example when ImageNet data will be downloaded.
  │
  ├── reports/                <- Directory to store reports/results/etc.
  │   └── figures             <- Generated graphics and figures to be used in reporting
  |
  ├── repo_util/              <- Folder containing all required artifacts for the README.md
```
### Getting started
To get started with the repo is really easy. Follow below steps to run everything by your self:
1. Open main notebook `EVA_Assignment-5.ipynb` and click on "open with cloab" option.
2. Run the first cell. This cell will git clone this repo so that all fucntions are available in your runtime.
3. That's it! Now you can execute the cell and train this three models. Other detials are commented in the main notebook.

## Type of Normalizations
Lets talk about Normalizations and what are different type of normalization.
#### Normalization
Normalization can be defined as a process where we try to zero center our data or we can say that we try to rescale our datapoints such that we have zero means and one standerd deviation. The process of normalization is also reversible i.e. we can revert back our data distribution from normalized state.
<br>
Below is an image which represent how normalization impact data points:
![Alt text](repo_util/normalize.JPG?raw=true "snippet")

A very simple intution behind doing normalization in Computer Vision based model architecture is to bring feature intensity to sample level. This also helps to maintain distributaion of feature map under certain limit and helps in smooth gradient decent.

#### Batch Normalization
- Batch Normalization is very common and most used Normalization technique when we develop Computer Vision models based on CNN.
- In this type of Normalization, we normalize channels in each layer accross batches. Means we calculate mean of each channel in layer accros batches. As shown in below image, we normalize same color channels across batches.
    ![Alt text](repo_util/batch_norm_work.JPG?raw=true "snippet")
- Book definitation of Batch Normalization is, "process to reduce internel coveriate shift" which means that Batch normalization reduces amplitude of similar features for example distribution of different dog noses.
- Batch normalization also reduces variance in loss while forward propogation and variance in gradient calculation while back propogation.
- Batch normalization parameter only depends on no. of channels in each layer and does not depend on batch size.

#### Layer Normalization
- Layer Normalization is common when we develop Natural Language Processing models based on RNNs.
- In this type of normalization, we normalize accross channels in a layer in each batch. Means we calculate mean across channels in each batch. As shown below, We have 4 batche and each batch has its mean and variance for that channel.
    ![Alt text](repo_util/layer_norm_work.JPG?raw=true "snippet")
- Layer normalization perfrom normalization across features and in NLP it gives euqual distributaion to each word in a sentance.
- Layer normalization parameters depend upon batch size hence it has alot of trainable parameters.

#### Group Normalization
- Group normalization is similar to Layer normalization in many aspect.
- In this instead of normalization across all channels like in layer, we divide the channels in groups and then normalize across that group. For example in below image, we have 4 channels in each image and we divided this channel into 2 groups and now we normalize across this group or similar color channels as shown in below image.
    ![Alt text](repo_util/group_norm_work.JPG?raw=true "snippet")
- Here the total number of parameters are euqal to number of channels time groups.

#### Normalization type together
![Alt text](repo_util/all_norm.JPG?raw=true "snippet")
In above image we can see all type of normalization. Here:
- N is images/datapoint in single batch.
- C represent channels/features in each image/datapoint
- H,W are height and width of channels.

1. Batch norm takes same channel across all images.
2. Layer norm takes all chanels across same image.
3. instance norm take single chanel across single image.
4. Batch norm takes group of channels acroos single image.
## Major modules
Here we will discuss on some of the major modules like dataset preperation module, model architecture and how to perform different normalization, Training module and how to perform L1 regularization.
<br>
Below are the sub topics and we will go through some important points.
#### Dataset preperation
First start with dataset module and how we are preparing dataset:
- We have a custom module `Dataset()` written in `components/data.py`. 
- This module downloads the data from torchvision and creates a iterable object.
- This module can also return dataloader of defined batch size.
- We can also pass transforms to this module as arguments. We have defined some in `components/transform.py`

#### Model architecture
Second we have custom module `Net()` to define model arcitecture:
- This module is written in `components/model.py`. Here we define our architecture.
- We have function to decide which normalization to perform as shown below. 
- Here we pass following for each normalization:
        - `BatchNorm2d` - output channel number of previous convolution layer.
        - `GroupNorm` - output channel number of previous convolution layer and group no. which is half of output channel.
        - `LayerNorm` - output shape in [C,H,W] of previous convoluation. (NOTE: elementwise_affine=False) to minimize parameters.
 ![Alt text](repo_util/model_snippet.JPG?raw=true "snippet")
#### Training and evaluation
We have custom module `TrainModel` defined in `training.py` which is responsible for executing forward and backward pass. In this module we have defined our **L1 regularization** in the loss function as shown below. It can be setup by passing Boolean while training.
![Alt text](repo_util/train_snippet.JPG?raw=true "snippet")

We also have custom module for testing/evaluating `TestModel` defined in `test.py`. Here we are saving misclassified images and labels for plotting them as shown in bleow snippet.
![Alt text](repo_util/test_snippet.JPG?raw=true "snippet")

## Model graphs
In this section, we will look into graphs generated by three models. Below is a snippet of model graph generated by our plot_util functions on calculated loss and accuracys.
##### Model 1 -> Model with Batch Norm + L1 regularization
##### Model 2 -> Model with Group Norm 
##### Model 3 -> Model with Layer Norm
![Alt text](repo_util/graph.JPG?raw=true "snippet")

## Misclassified images.
In this section we will look into 10 misclassified images from all three model as shown below:
#### First model (Batch norm + L1 regularization)
![Alt text](report/model1_misclassified.png?raw=true "snippet")
#### Second model (Group Normalization)
![Alt text](report/model2_misclassified.png?raw=true "snippet")
#### Third model (Layer Normalization)
![Alt text](report/model3_misclassified.png?raw=true "snippet")

## Conclusion
