DCFNet-Pytorch
============================
This repository is the pytorch implementation for ICML 2018 paper DCFNet: Deep Neural Network with Decomposed Convolutional Filters

For details please refer to [paper](https://arxiv.org/pdf/1802.04145.pdf).

The code for VGG borrows heavily from [torchvision](https://pytorch.org/docs/stable/torchvision/index.html).

**Docker on system Run**
```
docker start usman_math
nvidia-docker exec -it usman_math /bin/bash
```
How I run this;
```
nvidia-docker run -it --name usman_math -v /media/leo/0287D1936157598A/docker_ws/docker_ws:/app pytorch/pytorch:0.4.1-cuda9-cudnn7-runtime /bin/bash
```

**Access to the server**
```
ssh usman@ee4e099.ece.ust.hk
cd /data_shared/Docker/usman/tencent-ml-images
```

### Files
```
main-lasted.py : LAN file works VGG-DCF on MNIST Dataset

```
### Docker

#### Docker Image with Pytorch-0.4.1-cuda9-cudnn7

```sh
docker pull usmanmaqbool/math_dcf:pytorch-0.4.1-cuda9-cudnn7-runtime
sudo nvidia-docker run -it --name leo_math -v /media/leo/0287D1936157598A/docker_ws/docker_ws:/app -e DISPLAY=$DISPLAY -v /tmp/.X11-unix/:/tmp/.X11-unix/ usmanmaqbool/math_dcf:pytorch-0.4.1-cuda9-cudnn7-runtime /bin/bash
```