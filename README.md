# Security and Privacy of Machine Learning - Homework 2
> Black-box defenses on CIFAR-10 dataset

## Environments
### Training Environment
- Python version: `3.8.5`
- DL Framework: `PyTorch 1.7.0`
- Packages: Please refer to [requirements.txt](./requirements.txt)

#### Building Up the Environment
```bash
# Assume you already have pyenv :)
$ pyenv install 3.8.5
$ pyenv virtualenv 3.8.5 SPML-HW2
$ pyenv local SPML-HW2
$ pip3 install -r requirements.txt
```

### Inference Environment
- Python version: `3.8.5`
- DL Framework: `PyTorch 1.5.0`
- Packages: Please refer to [requirements-inference.txt](./requirements-inference.txt)

## Progress
- [ ] Paper survey for at least three methods
- [x] [Adversarial Training](https://arxiv.org/abs/1706.06083)
- [ ] Preprocessing-based defenses
    - SHIELD
