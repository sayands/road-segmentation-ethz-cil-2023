# Road Segmentation Project - Computational Intelligence Lab - ETHZ 2023

### Requirements 
Install a conda environment using ``requirements.yaml``. 
```bash
conda env create -f requirements.yaml
```

> If you're using GPU, the requirements file need to be updated for pytorch, since this was done on a MacOS.
### Usage

```bash
cd src
python train.py --config ../configs/base.yaml
```
