# CNN_model_in_Docker
Training a CNN model in Docker</br>

Commands</br>
Inside the projract directory</br>

```
docker build -f Dockefile -t cnn_model:v1 .
```
Once the image is successfully built, check if the image exists

```
docker images
```

To start the training

```
docker run cnn_model:v1 --batch_size 32 --epochs 10
```

