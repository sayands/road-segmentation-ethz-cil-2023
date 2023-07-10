# Road Segmentation Project - Computational Intelligence Lab - ETHZ 2023

### Requirements 
Install a conda environment using ``requirements.yaml``. 
```bash
conda env create -f requirements.yaml
```
### Usage

Change Paths in ``utils/define.py``

#### Craw GMAPS for curating dataset:
You need a valid Google maps API key set as environment variable: `export GMAPS_KEY=YOUR_API_KEY`
```bash
python data-preprocessing/crawl_aerial_seg.py
```


Training Script

```bash
cd src
python train.py --config ../configs/base.yaml
```

`config loss params` are as follows:
- `loss_type`: possible options are- ce (cross-entropy), dice ([dice loss](https://www.jeremyjordan.me/semantic-segmentation/#loss])), ftl ([focal tversky loss](https://towardsdatascience.com/dealing-with-class-imbalanced-image-datasets-1cbd17de76b5)), ce+dice (a combination of cross-entropy and dice loss) and ce+ftl (a combination of cross entropy and focal tversky loss)
- `wlambda`: activated when using ce+dice or ce+ftl loss_type. wlambda is the weightage given to cross-entropy loss. weightage of the other loss will be 1-wlambda.
- `alpha`: activated when using ftl or ce+ftl loss type. This is a hyperparameter required in focal tversky loss. The higher the alpha value, the more the false negatives are penalised. Weightage to false positives is 1-alpha.
- `gamma`: activated when using ftl or ce+ftl loss type. This is a hyperparameter required in focal tversky loss. Gamma is a parameter that controls the non-linearity of the loss. In the case of class imbalance, the FTL becomes useful when gamma > 1. Gamme < 1 is useful towards the end of training as the model is still incentivised to learn even though it is nearing convergence. 



Testing Script

```bash
cd src
python test.py --config ../configs/base_test.yaml
```

`config test params`:
- `test_path` : path to folder with test imgaes
- `mask_results_path` : path to where the mask images should be stored
- `submission_path` : path to submission file
- `model_path` : path to the model file
- `device` : cpu/cuda