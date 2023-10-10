# DelayNet


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
### Notice: 
We are using TensorFlowV2.11 in order to use Keras-TCN library. So, there are some expected issues installation.



# How to run
- Each `execute_*.sh` file is corresponding to each model
  - Example: `execute_model1_cnu.sh` is corresponding to `model1` experiment on CNU dataset 
    ```shell
    sh execute_model1_cnu.sh 
    ```
- Set Linux Commands to Run in the Background Using disown:
```shell
tmux new -d 'sh execute_model1_spain.sh > output.log'
```


