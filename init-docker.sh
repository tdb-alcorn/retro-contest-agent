#!/usr/bin/env bash

export DOCKER_REGISTRY=retrocontestbdesmkowhtvgpzax.azurecr.io
export DOCKER_USER=retrocontestbdesmkowhtvgpzax
docker login $DOCKER_REGISTRY \
    --username $DOCKER_USER \
    --password BG/waEpTCiFjzsTUIdMMk6HhMbZ9aUo9
