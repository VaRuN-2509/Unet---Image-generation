import json
import os
import torch
from torchvision import transforms
from torch.utils.data import Dataset

class polygonDataset(Dataset):
    def __init__(self,json_file,size=(256,256)):
        self.size = size
        self.transform_data = transforms.Compose([transforms.Resize(self.size),transforms.toTensor()])
        with open(json_file, 'r') as file:
            self.data = json.load(file)
            for item in self.data:
                image_file = os.path.join(os.path.dirname(json_file), item['input_image'])
                Image = Image.open(image_file).convert('RGB')
                item['input_image'] = self.transform_data(Image)
                print(f"Loaded image: {image_file} with shape {item['input_image'].shape}")
                Image.show()

data = polygonDataset('training\data.json', size=(256, 256))