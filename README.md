# CaVMamba



## 0. Main Environments

```bash
conda create -n cavmunet python=3.8
conda activate cavmunet
pip install torch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117
pip install packaging
pip install timm==0.4.12
pip install pytest chardet yacs termcolor
pip install submitit tensorboardX
pip install triton==2.0.0
pip install causal_conv1d==1.0.0  # causal_conv1d-1.0.0+cu118torch1.13cxx11abiFALSE-cp38-cp38-linux_x86_64.whl
pip install mamba_ssm==1.0.1  # mmamba_ssm-1.0.1+cu118torch1.13cxx11abiFALSE-cp38-cp38-linux_x86_64.whl
pip install scikit-learn matplotlib thop h5py SimpleITK scikit-image medpy yacs
```

The .whl files of causal_conv1d and mamba_ssm could be found here. {[Baidu](https://pan.baidu.com/s/1Tibn8Xh4FMwj0ths8Ufazw?pwd=uu5k)}

## 1. Prepare the dataset

### ISIC datasets

- The ISIC17 and ISIC18 datasets, divided into a 7:3 ratio, can be found here {[Baidu](https://pan.baidu.com/s/1Y0YupaH21yDN5uldl7IcZA?pwd=dybm) or [GoogleDrive](https://drive.google.com/file/d/1XM10fmAXndVLtXWOt5G0puYSQyI2veWy/view?usp=sharing)}. 

- After downloading the datasets, you are supposed to put them into './data/isic17/' and './data/isic18/', and the file format reference is as follows. (take the ISIC17 dataset as an example.)

- './data/isic17/'
  - train
    - images
      - .png
    - masks
      - .png
  - val
    - images
      - .png
    - masks
      - .png

### Synapse datasets

- For the Synapse dataset, you could follow [Swin-UNet](https://github.com/HuCaoFighting/Swin-Unet) to download the dataset, or you could download them from {[Baidu](https://pan.baidu.com/s/1JCXBfRL9y1cjfJUKtbEhiQ?pwd=9jti)}.

- After downloading the datasets, you are supposed to put them into './data/Synapse/', and the file format reference is as follows.

- './data/Synapse/'
  - lists
    - list_Synapse
      - all.lst
      - test_vol.txt
      - train.txt
  - test_vol_h5
    - casexxxx.npy.h5
  - train_npz
    - casexxxx_slicexxx.npz

## 2. Prepare the pre_trained weights

- The weights of the pre-trained VMamba could be downloaded [here](https://github.com/MzeroMiko/VMamba) or [Baidu](https://pan.baidu.com/s/1ci_YvPPEiUT2bIIK5x8Igw?pwd=wnyy). After that, the pre-trained weights should be stored in './pretrained_weights/'.



## 3. Train the CaVMamba

```bash
cd CaVMamba
python train.py  # Train and test CaVMamba on the ISIC17 or ISIC18 dataset.
python train_synapse.py  # Train and test CaVMamba on the Synapse dataset.
```

## 4. Obtain the outputs

- After trianing, you could obtain the results in './results/'

## 5. Acknowledgments

- We thank the authors of [VMamba](https://github.com/MzeroMiko/VMamba) and [Swin-UNet](https://github.com/HuCaoFighting/Swin-Unet)and VMUNet(https://github.com/JCruan519/VM-UNet) for their open-source codes.
