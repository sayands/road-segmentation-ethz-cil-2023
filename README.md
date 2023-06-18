# Road Segmentation Project - Computational Intelligence Lab - ETHZ 2023

:newspaper: No train validation split because we didn't discuss (put the provision in dataloader though)

### Requirements 

Partial requirements are added in:

```bash
conda env create --file environment.yml
```

if you want you can use pip to install them:

```bash
pip install -r requirements.txt
```


### Usage

Change Paths in ``utils/define.py``

#### Craw GMAPS for curating dataset:
You need a valid Google maps API key set as environment variable: `export GMAPS_KEY=YOUR_API_KEY`
```bash
python data-preprocessing/crawl_aerial_seg.py
```


Generate Image Fileset

```bash
python data-preprocessing/gen_fileset.py
```

See Dataloader

```
cd src
python src/datasets/aerial_data.py
```

If you want to visualise the loaded image and segmentation mask, go to the file and set ``visualise`` flag to True