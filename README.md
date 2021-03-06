# Security and Privacy of Machine Learning - Homework 2
> Black-box defenses on CIFAR-10 dataset

## Environments
### Training Environment
- Python version: `3.8.5`
- DL Framework: `PyTorch 1.7.0`
- Packages: Please refer to *[requirements.txt](./requirements.txt)*

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
- Packages: Please refer to *[requirements-inference.txt](./requirements-inference.txt)*.

#### Building Up the Environment
```bash
# Assume you already have pyenv :)
$ pyenv install 3.8.5
$ pyenv virtualenv 3.8.5 SPML-HW2-INF
$ pyenv local SPML-HW2-INF
$ pip3 install -r requirements-inference.txt
```


### Inference Time
I've tested my `hw2.py` on my PC. The runtime was within 10 seconds, and the GPU memory consumption
was about 1.5G.

### Test Results
> Update on 12/16 (Wed.)

- The test images and the corresponding answers are in *[test/](test/)*.
- I got **97**% accuracy on the test set!
- The `hw2_eval.py` is modified from `hw2.py` to evaluate the test set easier.
