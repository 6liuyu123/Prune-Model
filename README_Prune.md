### Preparation
```
    pip install nni==2.9
    pip install torchsummary
    conda install tensorboard
    mkdir ./pruned
```

### Prune Command
```
    python prune.py --modeltype audio(or image) --sparsity 0.3
    pruned model in ./pruned
```