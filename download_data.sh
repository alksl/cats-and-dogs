#!/usr/bin/env bash

mkdir -p data
cd data
kaggle competitions download -c dogs-vs-cats
unzip dogs-vs-cats.zip
unzip train.zip
unzip test1.zip

