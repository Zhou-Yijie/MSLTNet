# MSLTNet
WACV 2024 (Official implementation of "4K-Resolution Photo Exposure Correction at 125 FPS with ~ 8K Parameters")

## 
<figure class="half">
    <img src="https://github.com/Zhou-Yijie/MSLTNet/blob/main/fig1.jpg" width=100/>
    <img src="https://github.com/Zhou-Yijie/MSLTNet/blob/main/fig2.jpg" width=200>
</figure>


## Dependencies and Installation
```
# create new anaconda env
conda create -n mslt python=3.7 -y
conda activate mslt

# install python dependencies
pip3 install -r requirements.txt
```
## Prepare Dataset
Download ME Dataset from [Baidu Disk]() , unzip the file and put it in data/ 
## Quick Inference
```
# Inference On ME Dataset
python SingleTest.py
```
## Model Performance Testing
```
python test_model.py
```
## Train MSLT
```
python Train.py
```
