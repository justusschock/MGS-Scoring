import torch
import torchvision.models as models
import torchvision.transforms as transforms

import numpy as np
from PIL import Image
import os

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import json
#imagenet_classes = json.load(open('imagenet_classes.json'))
#idx2class = [imagenet_classes[str(i)].split(',')[0] for i in range(1000)]
#class2idx = {v:i for i,v in enumerate(idx2class)}

vgg16 = models.vgg16(pretrained=True)
vgg16.eval()
print(vgg16)

img_transforms = transforms.Compose([transforms.Scale((224, 224), Image.BICUBIC),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


def unnorm(x):
    un_x = 255*(x*0.5+0.5)
    return un_x.astype(np.uint8)


base_path = "/home/schock/Documents/mouse-project/segmentation"
img = Image.open('mouse.png').convert('RGB')
img_tensor = img_transforms(img)

img_var = torch.autograd.Variable(img_tensor.unsqueeze(0), requires_grad=True)
out = vgg16(img_var)
#print(str(np.argmax(out.data.numpy())) + ":" + idx2class[np.argmax(out.data.numpy())])

criterion = torch.nn.CrossEntropyLoss()
label = torch.autograd.Variable(torch.LongTensor(np.array([11])))
loss = criterion(out, label)
loss.backward()

fig = plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.imshow(unnorm(img_tensor.numpy()).transpose(1,2,0))
plt.subplot(1,2,2)
grads = np.abs(img_var.grad.data.numpy()[0]).max(axis=0)
plt.imshow(np.abs(img_var.grad.data.numpy()[0]).max(axis=0), cmap='gray')
fig.savefig(os.path.join(base_path, "test.png"))

img_seg = np.asarray(img.resize((224,224)))
seg_mask = np.zeros_like(img_seg)
seg_mask[np.where(grads> 0.15*np.amax(grads))] = 1

img_seg = img_seg * seg_mask

fig = plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.imshow(np.asarray(img.resize((224,224))))
plt.subplot(1,2,2)
plt.imshow(img_seg)
fig.savefig(os.path.join(base_path, "test_seg.png"))

print("")