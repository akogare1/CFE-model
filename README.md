### CFE Model: A Convolution-Based Explicit  Frequency-Aware Decomposition Model for Temporal Action Detection

#### Environment

- Python 3.8

- PyTorch >= 1.11

- NVIDIA GPU

  After installing the required packages, the following command needs to be executed:

  ```bash
  cd ./libs/utils
  python setup.py install --user
  cd ../..
  ```


#### Data Preparation

- We adpot the feature for **THUMOS14** and **ActivityNet-1.3** datasets from ActionFormer repository ([see here](https://github.com/happyharrycn/actionformer_release)). To use these features, please download them from their link and unpack them into the `./data` folder.

#### Started

##### Training

- Train our model with I3D features on **THUMOS14** dataset. This will create an experiment folder under *./ckpt* that stores training config, logs, and checkpoints.

```bash
python ./train.py ./configs/thumos_i3d.yaml --output reproduce
```

- Train our model with TSP features on **ActivityNet-1.3** dataset. This will create an experiment folder under *./ckpt* that stores training config, logs, and checkpoints.

```bash
python ./train.py ./configs/anet_tsp.yaml --output reproduce
```

#### Inference

We offer pre-trained models for each dataset, which you can download the checkpoint from [Google Driven](https://drive.google.com/drive/folders/1I9urS9TYXXJXlT3PZtVYR2bH-yKPTFrk?usp=drive_link). The command for test is

```bash
python eval.py ./configs/CONFIG_FILE PATH_TO_CHECKPOIN
```

