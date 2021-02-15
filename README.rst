Interaction Lab Data Processing and Modeling Pipeline
=====================================================


This modeling pipeline is meant to take you from data collection all the way through to using your model in the wild using data science best practices without an overwhelming amount of development required. With this pipeline you don't need to be a machine learning expert to create useful models of human behavior in the HCI and HRI domain.

Please see examples/walkthrough/ for a complete (step-by-step and end-to-end) walkthrough of this modeling process. 

## Philosophy
-----------------

We don't believe in reinventing the wheel and you shouldn't have to either. This pipeline is mostly a collection of external libraries cobbled together. Most of these are open and freely available, though some non essential elements are not. We try to always give credit where it is due, and we hope you will as well. 


# Getting Started
-----------------

We recommend you use a virtual environment such as conda for managing dependencies. We recommend use of python==3.5+

 - install all dependencies by running the following in the modeling-pipeline directory:

                `conda create \-\-name pipeline python=3.8`

                `conda activate pipeline`

                `pip install \-r requirements.txt`

Note: Due to hardware dependensies and install your version of `pytorch <https://pytorch.org/>`_

Pipeline Organization
===================

# Common
-----------------

In keeping with the python centric focus of this repository, we provide some basic utility functions for manipulating files in the file_utils.py script. These utilities are structured to either take a configuration file or src and dst pair. Most of these are simple wrappers on bash (TODO check compatability with windows) commands for moving files or manipulating media with ffmpeg. The most important however is get_dirs_from_config(), which produces a file list from a configuration file. This should work in most operating systems as long as the path is specified correctly in the config file. This is very helpful for working with datasets that can be organized idiosyncraticly.


# Feature Extraction
-----------------

Feature Extraction helps with taking audio and video and converting it into features that can be used for modeling. This is a mixture of wrappers around python libraries and seperate executables. 

 - `OpenFace <https://github.com/TadasBaltrusaitis/OpenFace>`_

 - `OpenSMILE <https://www.audeering.com/opensmile/>`_

 - `Librosa <https://librosa.org/doc/latest/index.html>`_

 - `Voice Activity <https://github.com/wiseman/py-webrtcvad>`_ (To json for annotation - with tool for annotation conversion)

 - `OpenPose <https://github.com/CMU-Perceptual-Computing-Lab/openpose>`_ [ToDo]

 - Cleaning OpenFace

 - Converting OpenPose to CSV [ToDo]


# Annotation
-----------------

 - Python Based Annotation

 - Prodigy [ToDo]

 - Inter-Rater Reliability Calculation [ToDo]

 - Converting Annotations to Labels


# Modeling
-----------------

This is based on PyTorch and sklearn.

 - Pick uncorrelated features

 - Data Loading

        \- High efficiency loading with feather

        \- External dataset configuration (mostly)

        \- Pytorch Dataset

                \- Test/Train/Val split

                \- Provides class weights

                \- Normalizes features

 - Model Definition

        \- Time series classification

        \- Text based Models [ToDo]

        \- Etc. [ToDo]

 - Training Scripts 

        \- Pytorch trainer

                \- Early stopping

                \- Complete metrics reporting

        \- Sklearn 

 - Hyperparameter Sweeps (Shown in examples)

        \- Bayesian optimization with Optuna

        \- Experiment tracking with Neptune


# New Sub Module Roadmap
-----------------

## Data Collection Examples
-----------------

Helpful examples for recording data during experiments for later use.

 - OBS Recording [ToDo]

 - ROSbag Recording [ToDo]

 - PSI [ToDo]


## Analysis
-----------------

[ToDo] Visualizing and analyzing results 

  - Visualize labels over test set [ToDo]

        \- SHAP <https://towardsdatascience.com/demystify-your-ml-model-with-shap-fc191a1cb08a>`_
  
  - Potential Integrations 

        \- Plot.ly

        \- GGplot

        \- Bokeh

        \- Se

        \- Gensim

## Production
-----------------

[ToDo] Using Models in the real world



# Contributing
==============

We welcome contributions. To contribute please make an issue requesting a change or a pull request with the expected change. To get started see the list below or any of the ToDo tags above.

- See [ToDo] tags above
- Improve citations
- Add Testing
- Autogen documentation
