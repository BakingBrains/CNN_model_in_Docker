FROM tensorflow/tensorflow:latest-gpu

WORKDIR ./docker_training

COPY . .

RUN apt-get update

#RUN pip install -r requirements.txt
RUN pip install matplotlib
RUN pip install scikit-learn
RUN pip install numpy
RUN pip install pillow

ENTRYPOINT [ "python3", "train.py" ]