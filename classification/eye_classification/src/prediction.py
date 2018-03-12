import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import torch
from torch.autograd import Variable
from skimage.measure import label, regionprops
from skimage import io

import regNet


data_transforms = transforms.Compose([
    transforms.CenterCrop(50),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

model = regNet.Net()

model.load_state_dict(torch.load('/home/temp/schneuing/Augen_Ergebnisse/' + 'Net[1.1207mae]_params.pth'))
model.eval()

def get_positions(image):
    """
    Computes center positions of image regions
    :param image:
    :return positions:
    """
    label_img = label(image)
    regions = regionprops(label_img)
    positions = []
    for props in regions:
        y, x = props.centroid
        positions.append((x, y))
    return positions

def get_eye_images(image, mask, len):
    images = []
    pos = get_positions(mask)
    for i, coord in enumerate(pos):
        x = coord[0]
        y = coord[1]
        img = image.crop((x-len/2, y-len/2, x+len/2, y+len/2))
        images.append(img)
    return images

dirname = '/home/temp/schneuing/Augen_Fotos/'
filename = 'W2_MGS_MVI_0297_154.png'
image = Image.open(dirname+filename)
mask = io.imread(dirname+filename.replace('.png', ' Kopie.png'),
                 as_grey=True)
images = get_eye_images(image, mask, 120)

# # Uncomment to get an image visualizing extracted eyes
# plt.subplot(1, len(images)+1, 1)
# plt.imshow(image)
# for i, img in enumerate(images):
#     plt.subplot(1, len(images)+1, i+2)
#     plt.imshow(img)
# plt.savefig('/home/students/schneuing/mouse-project/eye_classification/'+'eyes.png')

for i, img in enumerate(images):
    images[i] = data_transforms(img)

images = torch.stack(images, 0)

preds = model(Variable(images))
print(preds.data)

print('Prediction: {:.2f}'.format(torch.mean(preds.data)))
