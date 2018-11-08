#!/bin/bash

option=$1
NV_GPU=0
target_load=50
project_name=forge_load_gpu_${NV_GPU}

case "$option" in 
test)
  NV_GPU=${NV_GPU} nvidia-docker run -it --rm \
    yandi/forge_load:0.1-gpu bash
  ;;
stop)
  docker rm -f ${project_name}
  ;;
release)
  NV_GPU=${NV_GPU} nvidia-docker run -d\
    --log-driver=json-file --log-opt max-size=3m --log-opt max-file=3 \
    -e TARGET=${target_load} \
    --name ${project_name} \
    yandi/forge_load:0.1-gpu
  ;;
*)
  NV_GPU=${NV_GPU} nvidia-docker run -d\
    --log-driver=json-file --log-opt max-size=3m --log-opt max-file=3 \
    -e TARGET=${target_load} \
    --name ${project_name} \
    registry.api.weibo.com/forge_load/forge_load:0.1-gpu
esac
