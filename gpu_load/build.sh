docker build -t yandi/forge_load:0.1-gpu . -f Dockerfile
docker images | grep "^<none>" | awk '{print $3}' |
  while read -r image_id; do
    docker rmi $image_id;
  done
