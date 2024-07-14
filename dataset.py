import cv2
import pandas
from PIL import Image
from torch.utils.data import Dataset

class BirdDataset(Dataset):
  def __init__(self,csv_path,mode, transform=False) -> None:
    self.raw_csv = pandas.read_csv('/content/birds.csv')
    self.image_path = self.raw_csv[self.raw_csv['data set']==mode]
    self.image_path = self.image_path
    
    self.labels = self.image_path['labels'].unique()
    
    self.idx_to_class = {i:j for i,j in enumerate(self.labels)}
    self.class_to_idx = {j:i for i,j in enumerate(self.labels)}
    
    self.transform = transform
  
  def __len__(self):
    return len(self.image_path)

  def __getitem__(self, idx):
    image_filepath = self.image_path.iloc[idx,1]
    # this should be cwd + image path
    image = cv2.imread('/content/'+image_filepath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    label = self.image_path.iloc[idx,2]
    label = self.class_to_idx[label]
    if self.transform:
      image = self.transform(image=image)['image']
    return image, label