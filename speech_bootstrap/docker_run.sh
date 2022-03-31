#/usr/bin/bash

docker run -it \
           -v ${PWD}:/local/smart_drones \
           --gpus all \
           --rm vocal:latest /bin/bash
