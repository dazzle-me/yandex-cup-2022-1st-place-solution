# yandex-cup-2022-1st-place-solution
1st place solution of ML Audio Content track of Yandex Cup 2022

### Hardware
```
GPU: 1x4090
CPU: Ryzen 9 5950x
RAM: 128Gb
```
### Software
```
Python 3.8.10
CUDA: 11.8
torch build: torch==1.14.0.dev20221021+cu117
NVIDIA Driver Version: 520.56.06
```
### Directory structure
```
├── data
├── ensemble
├── ensemble.py
├── ensemble.sh
├── exps
├── prepare_env.sh
├── requirements.txt
├── train_arcface.py
├── train.sh
└── utils
```

### Data

Download and unzip data to ```./data/``` dir inside project root

### Install reqs

```bash prepare_env.sh ```

### Train model

It takes approximately 15 mins on the mentioned machine to finish one fold train, 
which is alone enough to take first place with score of 0.505-0.512 on public LB.

```bash train.sh```

### Inference
```bash ensemble.sh``` 

### Approximate tensorboard logs
Note: validation is done on one fold out of 10. 

![image](https://user-images.githubusercontent.com/57013219/201779620-31bb2e9e-3a99-45a3-af2e-c3a75ece93c0.png)
