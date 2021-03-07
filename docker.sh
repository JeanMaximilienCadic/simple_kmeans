docker rmi -f jcadic/simple_kmeans
docker rm jcadic-simple_kmeans
docker build . -t jcadic/simple_kmeans

docker run -it --rm -e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v $HOME/.Xauthority:/root/.Xauthority:rw jcadic/simple_kmeans
docker run -it --rm -e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v $HOME/.Xauthority:/root/.Xauthority:rw jcadic/simple_kmeans python -m simple_kmeans.visualization


