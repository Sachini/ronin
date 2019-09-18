# RoNIN: Robust Neural Inertial Navigation in the Wild

**Paper**: https://arxiv.org/abs/1905.12853  
**Website**: http://ronin.cs.sfu.ca/  
**Demo**: https://youtu.be/JkL3O9jFYrE

---
### Requirements
python3, numpy, scipy, pandas, h5py, numpy-quaternion, matplotlib, torch, torchvision, tensorboardX, numba, plyfile, 
tqdm, scikit-learn

### Data 
The dataset used by this project is collected using an [App for Google Tango Device](https://drive.google.com/file/d/1xJHZ_O-uDSJdESJhZ3Kpy86kWaGX9K2g/view) and an [App for any Android Device](https://drive.google.com/file/d/1BVhfKE6FEL9YRO1WQCoRPgLtVixDbHMt/view), and pre_processed to the data format specified [here](https://ronin.cs.sfu.ca/README.txt) 
Please refer to our paper for more details on data collection.

You can download the RoNIN dataset from our [project website](http://ronin.cs.sfu.ca/) or [HERE](https://www.dropbox.com/sh/wbfr1sl69i4npnn/AAADN9WorG3yYYnTlIK7iJWWa?dl=0). Unfortunately, due to security concerns we were unable to publish 50% of our dataset.

Optionally, you can write a custom dataloader (E.g: soure/data_ridi.py) to load a different dataset.

### Usage:
1. Clone the repository.
2. (Optional) Download the dataset from [HERE](https://www.dropbox.com/sh/wbfr1sl69i4npnn/AAADN9WorG3yYYnTlIK7iJWWa?dl=0) and the pre-trained 
models<sup>1</sup> from 
[HERE](https://www.dropbox.com/sh/3adl8zyp2y91otf/AABIRBecKwMJotMSrWE0z2n0a?dl=0). 
3. Position Networks 
    1. To train/test **RoNIN ResNet** model:
        * run ```source/ronin_resnet.py``` with mode argument. Please refer to the source code for the full list of command 
        line arguments. 
        * Example training command: ```python ronin_resnet.py --mode train --train_list <path-to-train-list> --root_dir 
        <path-to-dataset-folder> --out_dir <path-to-output-folder>```.
        * Example testing command: ```python ronin_resnet.py --mode test --test_list <path-to-train-list> --root_dir 
        <path-to-dataset-folder> --out_dir <path-to-output-folder> --model_path <path-to-model-checkpoint>```.
    2. To train/test **RoNIN LSTM** or **RoNIN TCN** model:
        * run ```source/ronin_lstm_tcn.py``` with mode (train/test) and model type. Please refer to the source code for the 
        full list of command line arguments. Optionally you can specify a configuration file such as ```config/temporal_model_defaults.json``` with the data
         paths.
        * Example training command: ```python ronin_lstm_tcn.py train --type tcn --config <path-to-your-config-file> 
        --out_dir <path-to-output-folder> --use_scheduler```.
        * Example testing command: ```python ronin_lstm_tcn.py test --type tcn --test_list <path-to-test-list> 
        --data_dir <path-to-dataset-folder> --out_dir <path-to-output-folder> --model_path <path-to-model-checkpoint>```.
4. Heading Network
    * run ```source/ronin_body_heading.py``` with mode (train/test). Please refer to the source code 
    for the full list of command line arguments. Optionally you can specify a configuration file such as 
    ```config/heading_model_defaults.json``` with the data paths.
    * Example training command: ```python ronin_body_heading.py train --config <path-to-your-config-file> 
    --out_dir <path-to-output-folder> --weights 1.0,0.2```.
    * Example testing command: ```python ronin_body_heading.py test --config <path-to-your-config-file> 
    --test_list <path-to-test-list>  --out_dir <path-to-output-folder> --model_path <path-to-model-checkpoint>```.

<sup>1</sup> The models are trained on the entire dataset. We will share the results from models trained on the public dataset (data lists available in 
_lists_ folder) for fair comaprison.

### Citation
Please cite the following paper is you use the code:  
[Yan, H., Herath, S. and Furukawa, Y. (2019). RoNIN: Robust Neural Inertial Navigation in the Wild: Benchmark, Evaluations, and New Methods. [online] arXiv.org. Available at: https://arxiv.org/abs/1905.12853](https://arxiv.org/abs/1905.12853)
