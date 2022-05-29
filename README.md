# Lunar Lander - Reinforcement Learning Study
Study of Reinforcement Learning solutions for the Lunar Lander environment.

## Install requirements

```
conda create --name lunar-lander-rl-study python=3.8
conda activate lunar-lander-rl-study
conda install swig
pip install -r requirements.txt
```

## Train

Train REINFORCE learning algorithm:

```
python agents/reinforce.py
```

## Evaluate

Run episodes for random agent:

```
python agents/random_action.py
```

Run episodes for trained REINFORCE model:

```
python -m evaluate
```
