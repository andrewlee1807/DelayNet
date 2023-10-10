# DelayNet: Enhancing temporal feature extraction for electronic consumption forecasting with delayed dilated convolution

This is the origin Tensorflow implementation of Informer in the following paper: [paper-link]

![img.png](imgs/model1.png) ![model2.png](imgs%2Fmodel2.png)
# Initialized environment (Linux OS)

```shell
# using Conda
conda env create -f environment.yaml  # Check the name of environment before import
conda activate ts_model
conda install -c conda-forge cudatoolkit=11.8.0
pip install nvidia-cudnn-cu11==8.6.0.163
CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/:$CUDNN_PATH/lib
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/:$CUDNN_PATH/lib' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh

```
# Requirements
Dependencies can be installed using the following command:
```shell
pip install -r requirements.txt
`````


### Notice: 
We are using TensorFlowV2.11 in order to use Keras-TCN library. So, there are some expected issues installation.

# Data
[Introduction data]

# Reproducibility
- Each `execute_*.sh` file is corresponding to each model
  - Example: `execute_model1_cnu.sh` is corresponding to `model1` experiment on CNU dataset 
    ```shell
    sh execute_model1_cnu.sh 
    ```
- Set Linux Commands to Run in the Background Using disown:
```shell
tmux new -d 'sh execute_model1_spain.sh > output.log'
```

# Usage

Commands for training and testing the model with any dataset:

```shell
python main.py \
    --write_log_file=True \
    --model_name="Model1" \
    --dataset_path="dataset/example.csv" \
    --config_path="dataset/example.yaml" \
    --output_length=$i \
    --device=0 \
    --output_dir="benchmark/exp/delay1_test"
```

We provide a more detailed and complete command description for training and testing the model:


```yaml
#INITIAL SETTINGS
kernel_size: 12 
gap: 24       # distance kernal mask
delay_factor: 3 # how many kernal mask refer to the past
nb_filters: 16  # Number of filters 
nb_stacks: 2 # Number of Delayed block, minimum=1
input_width: 168
train_ratio: 0.9
epochs: 10
optimizer: "adam"
metrics: [ 'mse', 'mae' ]
```

More parameter information please refer to `main.py`.

# Results

[Adding tables]


# Citation