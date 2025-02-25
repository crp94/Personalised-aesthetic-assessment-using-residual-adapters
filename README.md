[![DOI](https://zenodo.org/badge/148035757.svg)](https://zenodo.org/badge/latestdoi/148035757)
[![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/)


# Personalised aesthetics assessment in photography using deep learning and residual adapters
This repository contains the Jupyter notebooks used as supporting material for my msc thesis about personalised aesthetic assessment using residual adapters. The thesis was submitted in partial fulfillment of the requirements for the Msc in Artificial Intelligence at the University of Edinburgh in August 2018. A shorter version of this thesis was published as a preprint in [Arxiv](https://arxiv.org/abs/1907.03802) and presented as short talk in the [9th Iberian Conference on Pattern Recognition and Image Analysis](http://www.ibpria.org/2019/). Please visit the project in [Research Gate](https://www.researchgate.net/publication/335968544_Personalised_Aesthetics_with_Residual_Adapters), [Papers with code](https://paperswithcode.com/paper/personalised-aesthetics-with-residual), [Ground AI](https://www.groundai.com/project/personalised-aesthetics-with-residual-adapters/) and [Deep AI](https://deepai.org/publication/personalised-aesthetics-with-residual-adapters).

## Description

The notebooks in this repository were used to perform the experiments reported in my masters thesis on *Personalised aesthetics assessment in photography using deep learning*. This thesis had the aim of constructing deep learning models that could embed personalised aesthetic preferences on photography in deep learning models. To do so, we used **Residual adapters**, which have shown success in multi-domain deep neural networks. We argue that those adapters can learn the preferences of individual users over aesthetics in photography, regardless on what level of abstraction those preferences can represented in. They were presented in [this paper](http://homepages.inf.ed.ac.uk/hbilen/assets/pdf/Rebuffi17.pdf) and their optimal configuration was reported [here](http://homepages.inf.ed.ac.uk/hbilen/assets/pdf/Rebuffi18.pdf). To evaluate the performance of said adapters, we used the *Flicker-AES* dataset, and a similar train/test division as was used in [this report](http://users.eecs.northwestern.edu/~xsh835/assets/iccv2017_personalizedaesthetics.pdf), to make results comparable. 

### Experiments description

* First, we fine-tune a Resnet-18 network (pretrained on the Imagenet dataset), with the goal of predicting the preferences of the users of the train dataset. We argue that the preferences in this fine-tuned network can be used as a starting point to predict the preferences of the users in the test dataset. The resnet-18 architecture was chosen as it is quite lightweight compared to other architectures, such as Inception-V3 or deeper versions of the residual networks. This leads to faster training times, and we saw that deeper networks overfitted more often. The model trained in this experiment is used as the baseline for the rest of experiments. The trained network can be downloaded from [this link](https://drive.google.com/file/d/1030lZOL43_tWl0j8fXpzKO965ll1aQRj/view?usp=sharing).

* Second, we perform two distinct set of experiments in the test dataset. We assumed, in the first set of experiments, that we observe 10 ratings for each user in the test dataset. We re-train the bottleneck of the fine-tuned network using those 10 ratings and evaluate on the other images rated by each user (using cross-validation to make results more robust). The second set of experiments assume that we observe 100 ratings for each user in the test dataset. We try different approaches to train the network (re-train the bottleneck, fine-tune every layer in the network, and add the residual adapters and train them and the bottleneck, freezing all the other layers in the network). We found that fine-tuning every layer in the network leads to an increase in the model's capability similar to what was found using the adapters, but we argue that the use of a set of adapters for each user is preferrable, as it exploits parameter reuse, thus eliminating the need for one individual network for each user. 

* Third, we use the models above to perform a gradient-ascent-based photography enhancement method, so that the network can so be used to improve the expected aesthetics score of each picture by taking into account individual preferences. Additionally, we use Saliency Maps to understand what kind of features the network is looking at in the images, so we can further understand what features make some pictures more beautiful than others.

## Prerequisites
The notebooks require the installation of Python >= 3.5, Jupyter, PyTorch 1.0, CUDA >= 9.0 (for GPU usage), PIL, Pandas, Skimage, Scikit-learn, Matplotlib, Numpy and Scipy. This project was developed using Conda.

### Dataset

The dataset must be stored in a folder called "Images", which should be placed in the same directory as the notebooks. The images can be downloaded from the FLICKER-AES dataset, which can be found in [this repository](https://github.com/alanspike/personalizedImageAesthetics).

## Enhance your own picture
To enhance your own picture, save the network to a known location and execute the following line. You can control the intensity of the enhancement using --epsilon (we recommend values from 0.2 to 0.4 for best results). You need to have torch, PIL and numpy installed in your computer. Thanks to adaptive pooling, the network can enhance pictures of any size using the same neural network. 

```bash
$ python enhancepicture.py --epsilon 0.3 --network PATH_TO_THE_PRETRAINED_NETWORK --inputimage PATH_TO_YOUR_INPUT_IMAGE --outputimage DESIRED_PATH_FOR_THE_ENHANCED_PICTURE 
```

You can also enhance a list of pictures that are contained in the same directory. To do so, use the following script. The value of epsilon will be constant for all the images in the dictory. The enhanced images will be stored in the original directory with a prefix of "enhanced_". 

```bash
$ python enhancedirectory.py --epsilon 0.3 --network PATH_TO_THE_PRETRAINED_NETWORK --inputdirectory PATH_TO_YOUR_INPUT_DIRECTORY
``` 
Example of the result of the enhancement algorithm. I do not own the rights for the first picture. I took the second and third pictures  in the scottish highlands using a OnePlus 2 with HDR enabled. I also took the fourth picture in Madrid's Gran Vía using a Moto G6 Plus. 

Before enhancement         |  After enhancement
:-------------------------:|:-------------------------:
![](gettyimages-493656728.jpg)  |  ![](output.jpg)
![](IMG_20180818_131519.jpg)  |  ![](out1.jpg)
![](IMG_20180818_151222.jpg)  |  ![](out3.jpg)
![](IMG_d19xs3.jpg)  |  ![](out9.jpg)
## File descriptions

 * Every file ending in *.csv*: Datasets used to train and test the models
 * *MeanRatingFlickerAES resnet18.ipynb*: Notebook used to train and test the baseline resnet-18 model
 * *FineTuneEachWorker.ipynb*: Notebook used for the k=10 images/user setting
 * *FineTuneEachWorkerk100lastlayer.ipynb*: Notebook used for the k=100 images/user setting (re-training only the bottleneck of the network)
 * *FineTuneEachWorkerk100fullnetwork.ipynb*: Notebook used for the k=100 images/user setting (fine-tuning every layer of the network)
 * *Adapters.ipynb*:  Notebook used for the k=100 images/user setting (adding the adapters and training them and the bottleneck of the network)
 * *Saliency maps and picture enhancement.ipynb*: Code used for the personalised picture enhancement using gradient ascent and the saliency maps.
 * *Saliency maps and picture enhancement.ipynb-V2*: Code used for the personalised picture enhancement using gradient ascent and the saliency maps. This is an improved method of the previous notebook, now using the whole magnitude of the gradient to perform the image enhancement, and accepts networks with residual adapters as input.
 * *Dissertation.pdf*: Report containing the results of the experiments and other information. The thesis was submitted in partial fulfillment of the requirements for the Msc in Artificial Intelligence at the University of Edinburgh in August 2018. Selected as outstanding dissertation by the university.
 * *enhancepicture.py*: Script that automatically enhances one picture of any size. 
 * *enhancedirectory.py*: Script that automatically enhances every picture in a given directory.

  
## Authors

* **Carlos Rodríguez - Pardo** - [crp94](https://github.com/crp94)

If you find this work useful, please cite me and the paper in [Arxiv](https://arxiv.org/abs/1907.03802).

## License

This project is licensed under the GNU General Public License v3.0- see the [LICENSE.md](LICENSE.md) file for details
