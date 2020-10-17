import i2v4pytorch
from PIL import Image
import torchvision.transforms as T
import torchvision
import numpy as np
param_path = './models/i2v_pretrained_ver200_pytorch.bin'
tag_path = './tag_list.json'

illust2vec = i2v4pytorch.make_i2v_with_pytorch(param_path=param_path, tag_path=tag_path, mode='tag')

image_path = './images/miku.jpg'
img = Image.open(image_path)

# disable dropout
illust2vec.net.eval()

# estimate tags
print(illust2vec.estimate_plausible_tags([img], threshold=0.5))

# if you want to extract a semantic feature vector
illust2vec.change_mode(new_mode='feature')
print(illust2vec.extract_feature([img]))
