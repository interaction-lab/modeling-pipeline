Interaction Lab Data Processing and Modeling Pipeline
=====================================================


This modeling pipeline is meant to take you from data collection all the way through to using your model in the wild using data science best practices without an overwhelming amount of development required. With this pipeline you don't need to be a machine learning expert to create useful models of human behavior in the HCI and HRI domain.

Read the docs are now available in docs. Please see examples/walkthrough/ for a complete (step-by-step and end-to-end) walkthrough of this modeling process. 

Philosophy
-----------------

We don't believe in reinventing the wheel and you shouldn't have to either. This pipeline is mostly a collection of external libraries cobbled together. Most of these are open and freely available, though some non essential elements are not. We try to always give credit where it is due, and we hope you will as well. 


Getting Started
-----------------

We recommend you use a virtual environment such as conda for managing dependencies. We recommend use of python==3.5+

 - install all dependencies by running the following in the modeling-pipeline directory:

       `conda create \-\-name pipeline python=3.8`

       `conda activate pipeline`

       `pip install \-r requirements.txt`

Note: Due to hardware dependensies and install your version of `pytorch <https://pytorch.org/>`_



Contributing
==============

We welcome contributions. To contribute please make an issue requesting a change or a pull request with the expected change. To get started see the list below or any of the ToDo tags above.

- See [ToDo] Tags
- Improve Documentation and Citations
- Add Testing
- Features to Add:
       - OpenPose
       - Prodigy
       - Inter-Rater Reliability Calculator
       - Additional Model Types
- Modules to Add:
       - Data Recording
       - Analysis
       - Production
