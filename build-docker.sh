#!/usr/bin/env bash

agent=$1
echo "Preparing to build $agent..."
echo "  Hint: don't forget to update .dockerignore."
docker build -f docker/$agent.docker -t $DOCKER_REGISTRY/$agent:v1 .