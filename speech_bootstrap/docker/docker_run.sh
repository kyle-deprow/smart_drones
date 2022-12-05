#/usr/bin/bash

docker run -it \
           -v ${PWD}:/local/speech_bootstrap \
           --gpus all \
           --rm speech_bootstrap:latest /bin/bash
