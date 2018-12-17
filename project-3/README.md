# Project 3

You can follow the poster for the complete understanding of the project.

## Files

`Dectection-v5.ipynb` is the main file, where as `Transfer_CV_easy.ipynb` is for the transfer learning.

## Docker Configuration 

Basically it need python 2 and jupyter notebook with pytorch and sklean for running the code. Thats why we have used docker space for the configuration.

### To Start

```sh
xhost +local:root
# and start the images such as 

# Es image pr project final kar k commit kar k upload ki thi
nvidia-docker run -it --name glass_ws_p2 -v /media/leo/1E48BE700AFD16C7/docker_ws/Glass-Proj:/app -p 8003:8888 -e DISPLAY=$DISPLAY -v /tmp/.X11-unix/:/tmp/.X11-unix/ usmanmaqbool/glass_ws:finalproj-py2-jupyter /bin/bash


# Es image pe project complete kia tha
nvidia-docker run -it --name glass_ws_p2 -v /media/leo/1E48BE700AFD16C7/docker_ws/Glass-Proj:/app -p 8003:8888 -e DISPLAY=$DISPLAY -v /tmp/.X11-unix/:/tmp/.X11-unix/ usmanmaqbool/scatnet:server /bin/bash

```

### To Run

```sh
docker start glass_ws
docker exec -it glass_ws /bin/bash
cd /app/
jupyter notebook
```
After it ran, you can open it using [link](http://localhost:8003/)