# CoStudent
The codes for our work "Co-Student: Collaborating Strong and Weak Students for Sparsely Annotated Object Detection" based on [cvpods](https://github.com/Megvii-BaseDetection/cvpods.git)

## 1. Download resnet pre-trained model (Res-50,Res-101,ResX-101-32x8d) by url as follows:
[R-50](https://dl.fbaipublicfiles.com/detectron/ImageNetPretrained/MSRA/R-50.pkl)
[R-101](https://dl.fbaipublicfiles.com/detectron/ImageNetPretrained/MSRA/R-101.pkl)
[X-101-32x8d](https://dl.fbaipublicfiles.com/detectron/ImageNetPretrained/20171220/X-101-32x8d.pkl)

## 2. Prepare data

If you want to retrain our model on MS COCO, you need to ensure that the COCO dataset exist in your machine. Or you can head to [MS COCO](https://cocodataset.org/#download) to download the datasets.

Assume you have your COCO dataset in "/your-path/coco", and expected dataset structure as follows:
```
/your-path/coco/
            annotations/
            train2017/
            test2017/
            val2017/
```

link your COCO dataset to CoStudent root by:
```shell
(for Windows, need Administrator permissions) mklink /D "/path/CoStudent/datasets/coco" "/your-path/coco/" 
ln -s "/your-path/coco/" "/path/CoStudent/datasets/coco"
```

Download the sparse-annotations "missing_50p", "easy", "hard", and "extreme" from [Co-mining paper](https://drive.google.com/drive/folders/1jGl7IUxwJ3xRS0CcovzB7KEWMGZB555X?usp=sharing), which are publicly available. 
The annotation of "keep1" is from the authors of [SIOD paper](https://arxiv.org/abs/2203.15353) and [here](https://drive.google.com/drive/folders/1mJayvvNkmvur7IOG17-hz3AHQ2yPWfUf)


## 3. Prepare environment

**Step 1.** Create a conda environment and activate it.

```shell
conda create --name cvpods python=3.6 -y
conda activate cvpods
```

**Step 1.** Install corresponding version of torch and torchvision depends on the version of Cuda compilation tools you use.

Fisrt, check the version of Cuda compilation tools by:
```shell
nvcc -V
```
Assume your Cuda compilation tools vesion is 11.1,

If your machine is able to access the Internet, simply install using
```shell
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
```

Or, you can install them offline, download torch and torchvision by the following urls:
[torch]https://download.pytorch.org/whl/cu111/torch-1.8.0%2Bcu111-cp36-cp36m-linux_x86_64.whl
[torchvision]https://download.pytorch.org/whl/cu111/torchvision-0.9.0%2Bcu111-cp36-cp36m-linux_x86_64.whl

if your platform is Windows, download torch and torchvision by the following urls:
[torch]https://download.pytorch.org/whl/cu111/torch-1.8.0%2Bcu111-cp36-cp36m-win_amd64.whl
[torchvision]https://download.pytorch.org/whl/cu111/torchvision-0.9.0%2Bcu111-cp36-cp36m-win_amd64.whl

Upload these two whl files to your offline machine, and install them by:
```shell
pip install /path/torch-1.8.0%2Bcu111-cp36-cp36m-linux_x86_64.whl
pip install /path/torchvision-0.9.0%2Bcu111-cp36-cp36m-linux_x86_64.whl
```

<!-- **NOTE** (for Windows) [Build Tools for Visual Studio 2019 (version 16.9)](https://download.visualstudio.microsoft.com/download/pr/245e99d9-73d8-4db6-84eb-493b0c059e15/b2fd18b4c66d507d50aced118be08937da399cd6edb3dc4bdadf5edc139496d4/vs_BuildTools.exe) is needed. -->

**Step 2.** Install other needed packages
```shell
pip install -r requirements.txt
```

**Step 3.**  Build cvpods as follows:
```shell
cd /path/CoStudent
pip install -e .
```

**Step 4.** Training COCO
You can train our mothod CoStudent on COCO-miss50p for 12 epochs by the following command:
```bash
bash tools/train.sh
```








