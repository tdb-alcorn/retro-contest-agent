# OpenAI Retro Contest Agents

These are a collection of agents that I am using to compete in the OpenAI retro
gaming competition. Some of them are mine and some are taken from OpenAI's
baselines repositories. Also included are some scripts for generating large
datasets of training data, and Git LFS links to the datasets.


## Setup

To get set up, first follow all the instructions here: https://contest.openai.com/details


## Creating an agent

An agent is an interface, it must define four methods:

 1) `__init__(self)`: This is called once before any action begins. Here
    agent should set itself up, pulling configuration from `agents/config.py`.
 2) `load(self, sess, saver)`: This is called once before any action begins but after 
    the Tensorflow session has been initialized. Here the agent can load initial
    parameters from the filesystem from a checkpoint file specified in `agents/config.py`.
 3) `save(self, sess, saver)`: This can be called at any time by the training routine to
    save the updated parameters in a checkpoint file specified at `agents/config.py`. 
 4) `act(self, sess, state)`: This is called once every timestep
    _before_ the action happens. It should return an action that is valid for
    the environment.
 5) `step(self, sess, state, action, reward, next_state, done)`:
    This method is called once every timestep _after_ the action has
    happened in the environment. This is where the agent updates itself to account
    for changes in the environment and performs any online learning.
 6) `learn(self, sess, states, actions, rewards, next_states, episode_ends)`:
    This method is called during training by
    the trainer script. Each argument will be a numpy array of values.

Note that `sess` always refers to a Tensorflow session with global variables
already initialized.


## Training

This repo also defines two training mechanisms:

 1) `trainer.offline` is for offline, off-policy training. It simply runs
    batches of training data through the agents.
 2) `trainer.online` is for online, on-policy training. It runs agents through
    episodes of the task in a similar manner to the test environment.


## Submitting

Each agent defines its own docker file, found at `docker/<agent>.docker`. To test, 
first run `init_docker.sh` to set up the docker registry. Then run
`docker build -f <agent>.docker -t $DOCKER_REGISTRY/<agent>:v<X>` where <agent> is
the name of the agent to be tested and <X> is the next version number. You can check
what version number we're up to by running `docker pull`. Then go to https://contest.openai.com/user/job/submit
and use the docker tag <agent>:v<X> as the image name.


## Local testing

Before submitting a new agent you should test locally, which you can do by first
ensuring that your machine is set up according to the contest details page listed above.
Then simply run

```
retro-contest run --agent $DOCKER_REGISTRY/<agent>:v<X> \
	--results-dir results --no-nv --use-host-data \
	SonicTheHedgehog-Genesis GreenHillZone.Act1
```


## Generating training data

`trainer.generate` creates a large dataset of training examples by importing each
agent in turn and running it for some predetermined number of episodes. This should
generate a diverse range of training instances. Each training datum will be labeled
with the agent that generated it, the episode number and the frame number.

Generated data lives in the `data` directory, which should be checked in with git lfs
only.
