**Towards Zero Shot Learning in Restless Multi-armed Bandits**
==================================

We consider restless multi-arm bandits (RMABs), a multi-agent
reinforcement learning problem for resource allocation that has
applications in various scenarios such as healthcare, online adver-
tising, and anti-poaching. Previous works on RMABs mostly focus
on binary action and discrete state space and require training from
scratch when the system has new arms opt-in, which is common
in real-world public health applications. We develop a neural net-
work based pretrained model that has general zero-shot ability on a
wide-range of previously unseen RMABs, and can be fine-tuned on
specific instances in a more sample-efficient way than training from
scratch. Our model accommodates general multi-action and either
discrete or continuous state RMABs. The key idea is to use a new
model architecture utilizing feature information and use a novel
training procedure with streaming RMABs, to enable fast general-
ization. We derive a new updating rule for a crucial 𝜆-network with
theoretical convergence guarantees and empirically demonstrate
the advantage of our approach on several challenging real-world
inspired problems.

## Setup

Main file for PreFeRMAB, the main algorithm is `agent_oracle.py`

- Clone the repo:
- Install the repo:
- `pip3 install -e .`
- Create the directory structure:
- `bash make_dirs.sh`

Place data files mimiciii.csv, mimiciv.csv, and uganda.csv under folder 'uganda-rmab'

While installing mpi4py, the follow commands can be helpful:
sudo apt-get install libopenmpi-dev

To run Mimiciv dataset from the paper, run 
`bash run/job.run_mimiciv.sh`

Code adapted from https://github.com/killian-34/RobustRMAB, the github repo accompanying the paper "Restless and Uncertain: Robust Policies for Restless Bandits via Deep Multi-Agent Reinforcement Learning" in UAI 2023. 
