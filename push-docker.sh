#!/usr/bin/env bash

agent=$1
echo "Preparing to push $agent..."
echo "  Hint: evaluate locally with local-eval.sh before pushing."
docker push $DOCKER_REGISTRY/$agent:v1
echo "Push complete, go to https://contest.openai.com/user/job and submit a new job with tag \`$agent:v1\`"
