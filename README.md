# Interaction Lab Data Processing and Modeling Pipeline

This modeling pipeline is meant to take you from data collection all the way through to using your model in the wild using data science best practices without an overwhelming amount of development required. With this pipeline you don't need to be a machine learning expert to create useful models of human behavior in the HCI and HRI domain.

Please see our wiki [ToDo] for a complete (step-by-step and end-to-end) walkthrough of this modeling process. 

### Philosophy

We don't believe in reinventing the wheel and you shouldn't have to either. This pipeline is mostly a collection of external libraries cobbled together. Most of these are open and freely available, though some non essential elements are not. We try to always give credit where it is due, and we hope you will as well. 


## Getting Started

We recommend you use a virtual environment such as conda for managing dependencies. We recommend use of python==3.5+
 - install all dependencies with:

        `pip install -r requirements.txt`


# Script Organization

## 0. File Manipulation

Basic scripts for easily reogranizing data in folders and manipulating data files.

 - Changing framerates
 - Reorganizing files
 - Cropping videos
 - Changing file types or names
 - etc.

## 1. Data Collection Examples

Helpful examples for recording data during experiments for later use.

 - OBS Recording [ToDo]
 - ROSbag Recording [ToDo]

## 2. Features

### 2.1 Feature Extraction

 - OpenFace
 - OpenSMILE
 - Librosa
 - Voice Activity (To json for annotation)
 - OpenPose [ToDo]

### 2.2 Feature Conversion

 - Cleaning OpenFace
 - Converting OpenPose to CSV
 - Windowing Features [In-Progress]
 - Converting Annotations to Labels [ToDo]

### 2.3 Feature Importance

[ToDo] Methods for picking features to reduce correlation and redundancy

## 3. Annotation

 - Python Based Annotation
 - Prodigy [ToDo]
 - Inter-Rater Reliability Calculation [ToDo]

## 4. Modeling

 - Data Loading [In-Progress]
 - Model Definition
 - Training Scripts
 - Hyperparameter Sweeps

## 5. Analysis

[ToDo] Visualizing and analyzing results

## 6. Production

[ToDo] Using Models in the real world


# Additional Information

Here is a helpful video for [installing opensmile](https://www.youtube.com/watch?v=y8jDv1dW06Q&ab_channel=HowTo)



