### Installing DeepLabCut on a Geforce 610 in Ubuntu 16.04 LTS

- Install CUDA 8.0 (but not CUDNN 5.1)
```
wget https://developer.nvidia.com/compute/cuda/8.0/Prod2/local_installers/cuda-repo-ubuntu1604-8-0-local-ga2_8.0.61-1_amd64-deb
sudo dpkg -i cuda-repo-ubuntu1604-8-0-local-ga2_8.0.61-1_amd64.deb
sudo apt update
sudo apt install cuda
echo 'export PATH=/usr/local/cuda-8.0/bin${PATH:+:${PATH}}' >> ~/.bashrc 
exit
```
- Install a modfied CUDNN 5.1 to work on cards with compute capability < 3.0
- Install Python 3.6, Tensorflow 1.0, and DeepLabCut
```
conda create --name tf python=3.6
source activate tf
pip install --upgrade tensorflow-gpu==1.0
pip install deeplabcut
```
