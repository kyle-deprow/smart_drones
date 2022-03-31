#/usr/bin/bash

docker run -it \
           -v ${PWD}:/local/smart_drones \
           --gpus all \
           --rm subject:latest /bin/bash
