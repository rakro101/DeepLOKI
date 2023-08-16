# DeepLoki_

The vertical distribution of zooplankton in the ocean is patchy, and its relation to hydrographical conditions cannot be fully deciphered using traditional net casts due to the large depth intervals sampled. Optical systems that continuously take images during the cast can help bridge this gap. The Lightframe On-sight Keyspecies Investigation (LOKI) concentrates zooplankton with a net that leads to a flow-through chamber with a camera taking images with up to 20 frames sec$^{-1}$. 
These high-resolution images allow for the determination of zooplankton taxa, often even genera, species, and,  in case of copepods, developmental stages. 
Each cruise yields a large number of images, that need to be analyzed which is time consuming and currently requires internet access. 
To enhance the analyses, we developed a framework named DeepLoki, utilizing Deep Transfer Learning with various Convolution Neural Network Backbones. 
This work can be applied directly on board. We trained and validated the model on pre-labeled images from four cruises, while images from a fifth cruise were used for testing. 
Our best-performing model, based on the self-supervised pre-trained ResNet18 Backbone, achieved an average classification accuracy of 83.9\% which was high as compared to other approaches. 
In summary, we developed a tool for pre-sorting high-resolution black and white zooplankton images with high accuracy, which will simplify and quicken the final annotation process. 
In addition, we provide a user-friendly graphical interface for the DeepLoki framework for an efficient and concise processes leading up to the classification stage.

# Installation Guide
https://pytorch.org/get-started/locally/

```
brew install python@3.10
pip3 install torch torchvision torchaudio
pip3 install -r requirements_.txt
```

# Usage
First download the example_haul.zip, model_ckpt.zip and the sort.zip (get the content from the editors)
Extract them. 
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
