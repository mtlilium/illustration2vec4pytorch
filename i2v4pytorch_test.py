import i2v4pytorch
from PIL import Image

param_path = './models/i2v_pretrained_ver200_pytorch.bin'
tag_path = './tag_list.json'

illust2vec = i2v4pytorch.make_i2v_with_pytorch(param_path=param_path, tag_path=tag_path, mode='tag')
# print(illust2vec.net)

# image_path = './images/miku.jpg'
image_path = 'imgs/img_21.jpg'
# image_path = 'gyagu/001_1-1.png'
img = Image.open(image_path)


illust2vec.net.eval()
print(illust2vec.estimate_plausible_tags([img], threshold=0.5))
illust2vec.change_mode(new_mode='feature')
print(illust2vec.extract_feature([img])[0])
# print(illust2vec.net)
