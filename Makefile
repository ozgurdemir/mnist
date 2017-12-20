.PHONY: build run-it test train clean

TRAIN_SET_URL="http://mattmahoney.net/dc/text8.zip"

DIR=$(shell pwd)
DATA_DIR=$(DIR)/data
DOCKER_IMAGE="ozgurdemir/keras"
CONTAINER_NAME="word2vec-keras"
DOCKER_RUN=docker run --rm --name $(CONTAINER_NAME) -v $(DIR):/srv/ai -v $(DIR)/data:/root/.keras/datasets/ -w /srv/ai

build:
	docker build . -t $(DOCKER_IMAGE)
	mkdir -p data

run-it:
	$(DOCKER_RUN) -it $(DOCKER_IMAGE) /bin/bash

test:
	$(DOCKER_RUN) $(DOCKER_IMAGE) python3 -m unittest discover src

train:
	$(DOCKER_RUN) $(DOCKER_IMAGE) python3 src/train.py ${ARGS} --model data/model.hdf5 ${ARGS}

clean:
	rm data/*
