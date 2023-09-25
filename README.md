# A Multi-Modal Convolutional Neural Network Strategy for Spoof Detection

The method introduced aims to tackle the face anti-spoofing problem by leveraging a Convolutional Neural Network (CNN) approach while utilizing the WMCA dataset. This method utilizes a combination of color (C), depth (D), infrared (IR), and thermal (T) modalities to enhance the performance of the neural network model.


<div>
<img src="https://github.com/halashaker91/FAS/assets/122183607/d0c24747-b123-4985-917a-1eb8596c4164" width="300">
</div>


## The method of the proposed

The proposed method comprises two scenarios:

1. Pre-Fusion Scenario:
In this scenario, a series of experiments are conducted to assess the effectiveness of the system. Various combinations of multi-modal data are tested, and all modalities are swapped with each other to ensure comprehensive and realistic results.

<div>
<img src="https://github.com/halashaker91/FAS/assets/122183607/a16ebfc1-ffe8-461e-a57c-1d92484d990b" width="900">
</div>


2. Post-Fusion Scenario:
In this scenario, individual modalities are processed independently. Each modality undergoes a separate anti-spoofing algorithm, producing individual outputs. These outputs are then combined using different fusion methods, which include:
* Majority Voting.
* Weighted Voting.
* Averaging/Pooling.
* Stacking/Stacked Generalization.

  
<div>
<img src="https://github.com/halashaker91/FAS/assets/122183607/4d39415c-5080-414e-89ef-748f89a652d7" width="900">
</div>


### Requirements & Installation

You can install Spyder using the pip package manager, which is typically included with most Python installations. Prior to installing Spyder in this manner, you must first obtain the Python programming language from [the official Python website](https://www.python.org/).

Python >= 3.10.9 64-bit

To install Spyder and its other dependencies, You should run pip install spyder. 


#### Downloading the dataset

To access the WMCA database mentioned in the manuscript titled [Biometric Face Presentation Attack Detection with Multi-Channel Convolutional Neural Network](https://ieeexplore.ieee.org/abstract/document/8714076/), please click on the provided [link](https://zenodo.org/record/4580313). You will be directed to a page where you can enter your information and complete the agreement process for downloading the WMCA database.


##### Authors
Hala S. Mahmood, [Email](hala.shaker@uobasrah.edu.iq) , [Github](https://github.com/halashaker91)     
Salah Al-Darraji, [Email](aldarraji@uobasrah.edu.iq)‬
