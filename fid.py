import torch
import json
import os
import glob
from torchvision import transforms
from torchvision.transforms.functional import convert_image_dtype
from PIL import Image
device = "cuda"
_ = torch.manual_seed(123)
from torchmetrics.image.fid import FrechetInceptionDistance
import torchvision.transforms as T

# Formating neccessary functions
transform = T.Resize((299,299))
convert_tensor = transforms.Compose([
    transforms.PILToTensor()
])
fid = FrechetInceptionDistance(feature=64)
real = None
fake = None

# Process for bulk FID testing

# Read test groundtruth data
concepts_list = "/media/aioz-thang/data3/aioz-tuongdo/tune1.0/Test/concepts_list.json"
with open(concepts_list, "r") as f:
            concepts_list = json.load(f)
Ddirs = []
index = 1
for concept in concepts_list:
            Ddir = concept["instance_data_dir"]
            for real_lk in glob.glob(Ddir +"*"):
                Ddirs.append(real_lk)
            index += 1

# Read predicted data
predicted_folder = "/media/aioz-thang/data3/aioz-tuongdo/tune1.0/output-test200/"
prefix = "/*"

count = 0
print("Processing samples:")
for lk in range(1,len(Ddirs) + 1):
    temp_lk = predicted_folder + str(lk) + prefix 
    try:
        real = torch.cat((real,torch.unsqueeze(convert_tensor(transform(Image.open(Ddirs[lk-1]))),0)),0)
    except:
        real = torch.unsqueeze(convert_tensor(transform(Image.open(Ddirs[lk-1]))),0)
    for gen_lk in glob.glob(temp_lk):
        try:
            fake = torch.cat((fake,torch.unsqueeze(convert_tensor(transform(Image.open(gen_lk))),0)),0)
        except:
            fake = torch.unsqueeze(convert_tensor(transform(Image.open(gen_lk))),0)
    count  += 1
    print(str(count)+"/"+ str(len(Ddirs)))
print("Undercomputing FID...")
fid.update(real, real=True)
fid.update(fake, real=False)
print('Done! FID=',  str(fid.compute()))
