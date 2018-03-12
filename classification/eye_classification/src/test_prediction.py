import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import torch
from torch.autograd import Variable
from skimage import io
import math

import regNet
import helpers


data_transforms = transforms.Compose([
    transforms.CenterCrop(50),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

model = regNet.Net_Sure()
model = helpers.load_state_dict_part(model, torch.load('/home/temp/schneuing/Augen_Ergebnisse/'
                                                       + 'Net_Sure[1.0464mae]_params.pth'))
model.eval()

#######################################################################################################################
#  Test Images
#######################################################################################################################
image = Image.open('/home/students/schneuing/mouse-project/eye_classification/'+'blackimage.png').convert('RGB')
image = data_transforms(image)

pred, sure = model(Variable(image.unsqueeze(0)))

print('Prediction: {:.2f}, Sureness: {:.2f}'.format(pred.squeeze().data[0], sure.squeeze().data[0]))

#######################################################################################################################
#  Mouse Images
#######################################################################################################################
dirname = '/home/temp/schneuing/Augen_Fotos/'
filename = 'W2_MGS_MVI_0297_278.png'

image = Image.open(dirname+filename)
mask = io.imread(dirname+filename.replace('.png', ' Kopie.png'),
                 as_grey=True)
images = helpers.get_eye_images(image, mask, 120)
images_trafo = images.copy()
for i, img in enumerate(images_trafo):
    images_trafo[i] = data_transforms(img)

images_trafo = torch.stack(images_trafo, 0)

preds = model(Variable(images_trafo))

for i, img in enumerate(images):
    plt.subplot(math.ceil(len(images)/2), 2, i+1)
    plt.title('pred: {:.2f} ({:.2f} sure)'.format(preds[0].squeeze().data[i], preds[1].squeeze().data[i]))
    plt.imshow(img)
plt.savefig('/home/students/schneuing/mouse-project/eye_classification/'+'eye_results.png')



