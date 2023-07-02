from torch.utils.data import Dataset, DataLoader
from src import config as cfg
import os
import gzip
import numpy as np


class FashionDataset(Dataset):
    
    def __init__(self, data, transform = None):
        super().__init__()
        self.transform = transform
        self.labels = data[1]
        # Dimension of Images = 28 * 28 * 1. where height = width = 28 and color_channels = 1.
        self.images = data[0]
       

    def __getitem__(self, index):
        if self.transform == None:
            return self.images[index],self.labels[index]
        
        return self.transform(self.images[index]), self.labels[index]


    def __len__(self):
        return len(self.images)
    

	
def get_dataloader(data, transforms, batchSize, shuffle=True):
	# create a dataset and use it to create a data loader
	ds = FashionDataset(data,transforms)
	loader = DataLoader(ds, 
                    batch_size=batchSize,
		            shuffle=shuffle,
		            num_workers=1,
		            pin_memory=True if cfg.DEVICE == "cuda" else False,
                    pin_memory_device=cfg.DEVICE)
	# return a tuple of  the dataset and the data loader
	return (ds, loader)



def load_mnist(path, kind='train'):
    #load mnist data
    labels_path = os.path.join(path,kind+'-labels-idx1-ubyte.gz')
    images_path = os.path.join(path,kind+'-images-idx3-ubyte.gz')

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,offset=16).reshape(len(labels), 784)
    
    return images.reshape(-1,28, 28,1), labels



def split(data,frac=1):
    #split data into 2 parts
    assert frac <= 1
    indexes = np.arange(len(data[0]))
    np.random.shuffle(indexes)
    p1Len = int(np.floor(len(data[0])*frac))
    part1 = indexes[:p1Len]
    part2 = indexes[p1Len:]
    return (data[0][part1],data[1][part1] ), (data[0][part2],data[1][part2])


