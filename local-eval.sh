#!/usr/bin/env bash

retro-contest run --agent $DOCKER_REGISTRY/$1:v1 \
    --results-dir results --no-nv --use-host-data \
    SonicTheHedgehog-Genesis GreenHillZone.Act1
