## Imitation Learning (DAgger Algorithm)

This repository contains the code for an imitation learning model and the DAgger algorithm for the CarRacing-v0 Gym Environment. This was part of a homework assignment for the Deep Reinforcement Learning course at NYU Courant taught by [Lerrel Pinto](https://cs.nyu.edu/~lp91/).

![](https://github.com/kvgarimella/dagger/blob/main/media/dagger.gif)

DAgger helps the imitation learning agent learn correct actions when following sub-optimal trajectories:

![](https://github.com/kvgarimella/dagger/blob/main/media/self-correction.gif)

Check out [this paper](https://www.cs.cmu.edu/~sross1/publications/Ross-AIStats11-NoRegret.pdf) to learn more about DAgger. 

### Installation
Clone this repository:
```
git clone https://github.com/kvgarimella/dagger.git
cd dagger
```
Install the requirements:
```
pip install -r requirements.txt
```
Run DAgger and train your model:
```
python dagger.py
```


