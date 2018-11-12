### Software
- Miniconda
	- https://conda.io/miniconda.html
- CUDA and cuDNN (if building from source)
	- https://developer.nvidia.com/cuda-92-download-archive
	- https://developer.nvidia.com/rdp/cudnn-download
- PyTorch
	- https://pytorch.org/
	- https://github.com/pytorch/pytorch
- PyCharm
	- https://www.jetbrains.com/pycharm/
	- https://www.jetbrains.com/student/
- Netron
	- https://github.com/lutzroeder/netron

### Installation
Recommended operating systems are Ubuntu 16.04, Windows 10, or macOS 10.13 (Hih Sierra).

1. Create a new conda environment using your terminal (Linux or macOS) or the anaconda prompt (Windows), and activate the environment.
```
conda create --name pytorch python=3.6
source activate pytorch
```

2. Select the correct binary from https://pytorch.org/, for example the Linux binary for CPU support can be installed using:
```
conda install pytorch torchvision -c pytorch
```

3. Install the PyCharm IDE and setup a new project. You'll have to import the conda environment before you can use it.

### PyTorch examples
The example from the recitation is contained within ```example-1.py```, an example using the actual AlexNet and CaffeNet values is contained within ```example2.py```.
