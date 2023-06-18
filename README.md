# Road Segmentation Project - Computational Intelligence Lab - ETHZ 2023

### Requirements 
Install a conda environment using ``requirements.yaml``. 
```bash
conda env create -f requirements.yaml
```

> If you're using GPU, the requirements file need to be updated for pytorch, since this was done on a MacOS.
### Usage

Change Paths in ``utils/define.py``

#### Craw GMAPS for curating dataset:
You need a valid Google maps API key set as environment variable: `export GMAPS_KEY=YOUR_API_KEY`
```bash
python data-preprocessing/crawl_aerial_seg.py
```


Generate Image Fileset

```bash
cd src
python train.py --config ../configs/base.yaml
```
