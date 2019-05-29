# RoNIN: Robust Neural Inertial Navigation in the Wild

**Paper**: coming soon\
**Website**: http://ronin.cs.sfu.ca/\
**Demo**: https://youtu.be/JkL3O9jFYrE

---
### Requirements
python3, numpy, scipy, pandas, h5py, numpy-quaternion, matplotlib, torch, torchvision, tensorboardX, numba, plyfile, 
tqdm, scikit-learn

### Data format
The dataset used by this project is collected using an [App for Google Tango Device](https://drive.google.com/file/d/1xJHZ_O-uDSJdESJhZ3Kpy86kWaGX9K2g/view) and an [App for any Android Device](https://drive.google.com/file/d/1BVhfKE6FEL9YRO1WQCoRPgLtVixDbHMt/view), and pre_processed to the data format specified [here](http://ronin.cs.sfu.ca/README.txt) 
Please refer to our paper for more details on data collection.

You can download the RoNIN dataset from our project website. Optionally, you can write a custom dataloader (E.g: soure/data_ridi.py) to load a different dataset.

### Usage:
1. Clone the repository.
2. (Optional) Download the dataset from [HERE](http://ronin.cs.sfu.ca/) and the pre-trained model from [HERE]
(coming_soon). 
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
        * Example testing command: ```python ronin_lstm_tcn.py test type lstm_bi --test_list <path-to-test-list> 
        --data_dir <path-to-dataset-folder> --out_dir <path-to-output-folder> --model_path <path-to-model-checkpoint>```.
4. Heading Network
        * run ```source/ronin_body_heading.py``` with mode (train/test). Please refer to the source code 
        for the full list of command line arguments. Optionally you can specify a configuration file such as 
        ```config/heading_model_defaults.json``` with the data paths.
        * Example training command: ```python ronin_body_heading.py train --config <path-to-your-config-file> 
        --out_dir <path-to-output-folder> --weights 1.0,0.2```.
        * Example testing command: ```python ronin_body_heading.py test --config <path-to-your-config-file> 
        --test_list <path-to-test-list>  --out_dir <path-to-output-folder> --model_path <path-to-model-checkpoint>```.

### Citation
Please cite the following paper is you use the code: coming soon