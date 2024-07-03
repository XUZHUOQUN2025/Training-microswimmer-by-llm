# Code for Training microrobots to swim by a large language model

[This repository]([https://www.google.com](https://github.com/ZhulailaiFluidLab/Training_microswimmer_by_LLM)) contains code of the following paper:

Xu Z, Zhu L. Training microrobots to swim by a large language model[J]. arXiv preprint arXiv:2402.00044, 2024. [arxiv](https://arxiv.org/abs/2402.00044)


## Project dependencies

Ensure your Python version is at least 3.11.4. Install the required libraries for this project:

```bash
pip install openai==0.28
pip install numpy
pip install retrying
```

## Select the swimmer to control

```bash
swimmer = ThreeLink() # for the purcell swimmer
swimmer = ThreeSphere() # for NG's swimmer (line 803)
```

## Change swimmer's moving direction (positive or negative)

```text
When you want to change the moving direction (positive or negative):
1. Change the first sentence of the prompt (see line 766).
2. Modify the history-clear section (see lines 783-790).
```
## Change the number of historical demonstrations

```bash
self.history_length = xxx # (line 684)
```


<!--There are two class "class ThreeLink" and "class Threesphere" respectively introduce the environment of purcell swimmer and NG's swimmer.   -->
