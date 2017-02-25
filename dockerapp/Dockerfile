FROM continuumio/miniconda
RUN pip install theano
RUN conda install tensorflow
RUN pip install Keras==1.1.2
RUN pip install joblib==0.10.3
RUN pip install h5py
RUN mkdir /ny
WORKDIR /ny
ADD . /ny

