import json
import os
import sys
import torch
from matplotlib.colors import to_rgb
from PIL import Image
import torch.nn as nn
import torch.optim as optim
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Unet_Model.Unet import Unet
from torchvision import transforms

class polygonDataset():
    def __init__(self,json_file,size=(256,256)):
        self.size = size
        self.transform_data = transforms.Compose([transforms.Resize(self.size),transforms.ToTensor()])
        with open(json_file, 'r') as file:
            self.data = json.load(file)
            for item in self.data:
                image_file = os.path.join(os.path.dirname(json_file), 'inputs',item['input_polygon'])
                img = Image.open(image_file).convert('RGB')
                item['input_image'] = self.transform_data(img)
                item['input_colour'] = self.colour_tensor(item['colour'])
                item['output_image'] = self.transform_data(Image.open(os.path.join(os.path.dirname(json_file), 'outputs', item['output_image'])).convert('RGB'))
                #print(f"Loaded image: {image_file} input colour {item['colour']} {item['input_colour'].shape} with shape {item['input_image'].shape}")

    def colour_tensor(self,colour):
        if isinstance(colour, str):
            self.colour_ = to_rgb(colour)
            #print(f"Converted colour {colour} to RGB tensor {self.colour_}.")
        else:
            raise ValueError("Colour must be a string representing a color name or hex code.")
        return torch.tensor(self.colour_, dtype=torch.float32).view(3, 1, 1)

    def len(self):
        return len(self.data)
    
    def getitem(self, idx):
        item = self.data[idx]
        return {
            'input_image': item['input_image'],
            'input_colour': item['input_colour'],
            'output_image': item['output_image']
        }
                

def loss_fn(len_data):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    n_epochs = 10
    model.train()
    for n in range(n_epochs):
        running_loss = 0.0
        for i in range(len_data):
            item = data.getitem(i)
            input_image = item['input_image'].unsqueeze(0).to(device)
            B,C,H,W = input_image.shape
            input_colour = (item['input_colour'].expand(3,H,W)).unsqueeze(0).to(device)
            target = item['output_image'].unsqueeze(0).to(device)
            input_embed = torch.cat((input_image, input_colour), dim=1)
            print(f"Input shape: {input_embed.shape}, Target shape: {target.shape} for input {i}" )
            output = model.forward(input_embed).to(device)
            
            optimizer.zero_grad()
            loss = nn.L1Loss()(output, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        print(f"Epoch {n+1}/{n_epochs}, Loss: {running_loss/len_data:.4f}")

if __name__ == '__main__':
    input_image = torch.rand((1, 3, 512, 512))
    model = Unet()
    # Total parameters and trainable parameters.
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters.")
    data = polygonDataset('training\data.json', size=(512, 512))
    print(f"Dataset length: {data.len()}")
    print("starting training...")
    loss_fn(data.len())
    