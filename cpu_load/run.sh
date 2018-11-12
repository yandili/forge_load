#!/bin/bash

option=$1
target_load=55
project_name=forge_load_cpu

case "$option" in 
test)
  docker run -it --rm \
    yandi/forge_load:0.1-cpu bash
  ;;
stop)
  docker rm -f ${project_name}
  ;;
release)
  docker run -d\
    -e TARGET=${target_load} \
    --name ${project_name} \
    yandi/forge_load:0.1-cpu
  ;;
*)
  docker run -d\
    -e TARGET=${target_load} \
    --name ${project_name} \
    registry.api.weibo.com/forge_load/forge_load:0.1-cpu
esac
