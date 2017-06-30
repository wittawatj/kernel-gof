# Set the base image

# Dockerfile at https://github.com/ContinuumIO/docker-images/blob/master/miniconda/Dockerfile
FROM continuumio/miniconda

RUN apt-get update --fix-missing
RUN apt-get install -y gcc
RUN pip install jupyter

# install kgof from https://github.com/wittawatj/kernel-gof
RUN pip install git+https://github.com/wittawatj/kernel-gof.git

MAINTAINER Wittawat Jitkrittum <wittawatj@gmail.com>

