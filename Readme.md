#OVERVIEW

This project creates a machine learning pipeline which sends an image as an HTTP POST request to a server which contains a pre-trained classification model(DenseNet-121) and returns the classification result. The entire image classification pipeline is hosted in a docker image, which can run across any docker container, making it platform independent.

#INSTRUCTIONS

Please install docker in your machine before running the following instructions. In the repository, we have a Dockerfile, which is a text file and can be used to create a docker image. To do this, we can run the following command in the terminal:
```
docker build -t <Docker Image Name>:<Version> .
```
Now, the docker image is created and can be run in a docker container. To do this, run the following command in the terminal:
```
docker run -p 5000:5000 <Docker Image Name>:<Version>
```
This will start a docker container which will have the docker image running on it. Our app will be hosted on localhost (http://127.0.0.1:5000) on the port 5000. Also, we are using HTTP POST requests to receive the images. To send an image to the docker app, use the following curl command:
```
curl -F "file=@<Your Image File Name>"  http://127.0.0.1:5000/
```
Here, the image is a locally stored image on your machine. An example request is:
```
curl -F "file=@dog.jpg"  http://127.0.0.1:5000/
```
where "dog.jpg" is an image stored in the current directory. This will return a label corresponding to the classification result.
