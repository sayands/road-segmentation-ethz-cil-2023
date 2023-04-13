# Road Segmentation Project - Computational Intelligence Lab - ETHZ 2023

:newspaper: No train validation split because we didn't discuss (put the provision in dataloader though)

### Requirements 

Will put later!
### Usage

Change Paths in ``utils/define.py``

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