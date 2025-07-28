#! /bin/bash

mkdir data
cd data
wget http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat
mkdir bin

# ⬇️ Baixa o COCO 2014 (treinamento)
wget http://images.cocodataset.org/zips/train2014.zip

# 🗜️ Extrai o conteúdo
unzip -q train2014.zip