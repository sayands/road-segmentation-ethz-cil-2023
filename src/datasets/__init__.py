from .aerial_data import AerialSegCustom

def get_dataset(dataset_name):
    if dataset_name == 'CrawlSeg':
        return AerialSegCustom
    else:
        raise NotImplementedError