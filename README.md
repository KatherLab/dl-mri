# dl-mri
Deep Learning MRI image analysis  
Core codes of tensorflow implementation for end to end survival prediction in rectal cancer MRI

### Data
A csv file in the following formate, see also [clini.csv](./clini.csv):

|id | ori_path      | mask_path                | os | os_e | dfs | dfs_e |
|--- |---------------|--------------------------| --- | --- | --- | --- |
|10000 | /input/ori/A1 | ../input/mask/pre-A1.nii | 100 | 1 | 100 | 1 |


### Training

Training script is in [train.py](./train.py) file, network structure based on ViTB/16,
from [ViT-kera](https://github.com/faustomorales/vit-keras) repository,
a Keras implementation of ViT.

### Loss

Cox partial likelihood as the [loss](./loss.py) function for the network;
Cindex is the evaluation metric of model.  
$$L(\theta)= -\sum_{i: E_{i}=1}\left(\hat{h}_{\theta}\left(x_{i}\right)-\log \sum_{j \in \Re\left(T_{i}\right)} e^{\hat{h}_{\theta}\left(x_{j}\right)}\right)$$