# Instructions to Run the App:

docker ps

Note the `container id`

docker stop <container id>

docker rm <container id>

cd `/path/to/project/root`

docker build --tag python-docker .

docker run -d -p 5000:5000 python-docker
