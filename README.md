# DeepLOKI

Zooplankton plays a crucial role in the ocean’s ecology, serving as a foundational component in
the food chain by consuming phytoplankton or other zooplankton and furthermore influencing
nutrient cycling. This pivotal role distinguishes them from other species that reside at higher
trophic levels. The vertical distribution of zooplankton in the ocean is patchy, and its relation
to hydrographical conditions cannot be fully deciphered using traditional net casts due to the
large depth intervals sampled. Optical systems that continuously take images during the cast
can help bridge this gap. The Lightframe On-sight Keyspecies Investigation (LOKI) concentrates
zooplankton with a net that leads to a flow-through chamber with a camera taking images with
up to 20 frames sec−1. These high-resolution images allow for the determination of zooplankton
taxa, often even to genus or species level, and, in the case of copepods, developmental stages.
Each cruise produces a substantial volume of images, ideally requiring onboard analysis, which
presently consumes a significant amount of time and necessitates internet connectivity to access
the EcoTaxa Web service. To enhance the analyses, we developed an AI-based software
framework named DeepLOKI, utilizing Deep Transfer Learning with a Convolution Neural Network
Backbone. Our DeepLOKI image recognition tool can be applied directly on board. We trained
and validated the model on pre-labeled images from four cruises, while images from a fifth
cruise were used for testing. The best-performing model, utilizing the self-supervised pre-trained
ResNet18 Backbone, achieved a notable average classification accuracy of 83.9 %, surpassing
the regularly and frequently used method EcoTaxa (default) in this field by a factor of two. 
In summary, we developed a tool for pre-sorting high-resolution black and white zooplankton images
with high accuracy, which will simplify and quicken the final annotation process. In addition, we
provide a user-friendly graphical interface for the DeepLOKI framework for efficient and concise
processes leading up to the classification stage. Moreover, performing latent space analysis on
the self-supervised pre-trained ResNet18 Backbone could prove advantageous in identifying
anomalies such as deviations in image parameter settings. This, in turn, enhances the quality
control of the data. Our methodology remains agnostic to the specific imaging end system used,
such as Loki, UVP, or ZooScan, as long as there is a sufficient amount of appropriately labeled
data available to enable effective task performance by our algorithms.

# Installation Guide
https://pytorch.org/get-started/locally/

pip install torch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1

```
brew install python@3.10
pip3 install torch torchvision torchaudio
pip3 install -r requirements_.txt
```

# Usage
First download the example_haul.zip, model_ckpt.zip, loki.zip. and the sort.zip from
[Download Here](https://uni-duesseldorf.sciebo.de/s/okWh4728VwnCBGp).
Extract them.
loki folder in the DeepLoki_ folder on root level. (here are our models stored)

Copy the update_allcruises_df_validated_5with_zoomie_20230727.csv to output.
Copy the update_wo_artefacts_test_dataset_PS992_20230727_nicole.csv to output.

Copy the example_haul folder to data/ .
Copy the sort folder to data/5_cruises/ .
Copy the content to saved_models/ .

Image analysis: Run start_app.py

Image Labeling: Run start_app_sort.py

# Training - Data needed and computing power

Training: Run train_pytorch_lightning_model.py

PreTraining: Run pretrain/pretrain_with_dino_paper_resnet_dino450.py

# Software used
Training and Validation was performed on an Nvidia A$100$ (Nvidia Corp., Santa Clara, CA, USA) and on Apple M1 MAX with 32 GB (Apple, USA), depending on the computational power needed, for example self-supervised pre-training was performed on a Hyper performing cluster with Nvidia A$100$. <br>
On the Macbook Pro (Apple, USA) we used:<br>
Python VERSION:3.10.5<br>
pyTorch VERSION:13.1.3<br>
On the cluster we used cluster specifics versions of the software:<br>
Python VERSION:3.10.5 <br>
pyTorch VERSION:13.1.3<br>
CUDNN VERSION:1107)<br>

# Authors
Raphael Kronberg and Ellen Oldenburg

# Support 
If you **really** like this repository and find it useful, please consider (★) **starring** it, so that it can reach a broader audience of like-minded people. It would be highly appreciated !

# Contributing to DeepLOKI
If you find a bug, create a GitHub issue, or even better, submit a pull request. Similarly, if you have questions, simply post them as GitHub ([Link text Here](https://github.com/rakro101/DeepLOKI)) issues. 


# License , citation and acknowledgements
Please advice the **LICENSE.md** file. For usage of third party libraries and repositories please advise the respective distributed terms. Please cite our paper, when using this code:

```
@software{kronbergapplicationsdeeploki,
  title={DeepLOKI- A deep learning based approach to identify Zooplankton taxa on high-resolution images from the optical plankton recorder LOKI},
  author={Kronberg, Raphael Marvin and Oldenburg, Ellen}
  year = {2023},
  url = {https://github.com/rakro101/DeepLOKI},
}
```
