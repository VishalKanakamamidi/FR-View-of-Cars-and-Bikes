import matplotlib.pyplot as plt

import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from PIL import Image
from torch.autograd import Variable
import cv2
import os
import random

test_transforms = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224),transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model=torch.load('/content/drive/MyDrive/Cars_Bikes/model.pt')# Model path
model.eval()
def predict_image(image):
	image_tensor = test_transforms(image).float()
	image_tensor = image_tensor.unsqueeze_(0)
	input = Variable(image_tensor)
	input = input.to(device)
	output = model(input)
	index = output.data.cpu().numpy().argmax()
	return index

def predict_show_image(image_name):
	im = cv2.imread(image_name)
	im = Image.fromarray(im)
	plt.imshow(im)
	res = predict_image(im)
	if(res==1):
		ch = "Front_Car"
	if(res==0):
		ch = "Front_Bike"
	if(res==2):
		ch = "Rear_Bike"
	if(res==3):
		ch = "Rear_Car"

	print("The class of input image is ",ch)
	plt.title(ch)
	plt.show()




image_name = "car.jpg" # input image name here
predict_show_image(image_name)




