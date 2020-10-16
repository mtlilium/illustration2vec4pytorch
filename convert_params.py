from i2v4pytorch.i2v_net import I2VNet
import torch
import numpy as np
'''
Original i2v code
'''
###

#import i2v
#from chainer import serializers

# illust2vec_feature = i2v.make_i2v_with_chainer('illust2vec_ver200.caffemodel')
# illust2vec_tag = i2v.make_i2v_with_chainer('illust2vec_tag_ver200.caffemodel')
# serializers.save_npz('models/i2v_ver200.npz',illust2vec_feature.net)
# serializers.save_npz('models/i2v_tag_ver200.npz',illust2vec_tag.net)

###

def exchange_param_name(key):
    key = key.split('/')
    if key[1] == 'W':
        key[1] = '.weight'
    elif key[1] == 'b':
        key[1] = '.bias'
    return ''.join(key)

net = I2VNet()
print(net)

tag_premodel = np.load('models/i2v_tag_ver200.npz')
feature_premodel = np.load('models/i2v_ver200.npz')

intersection = set(tag_premodel.files) & set(feature_premodel.files)

for key in intersection:
    net.state_dict()['block.' + exchange_param_name(key)][0:] = torch.from_numpy(tag_premodel[key]).float()

net.state_dict()['tag_classifier.conv6_4.weight'][0:] = torch.from_numpy(tag_premodel['conv6_4/W']).float()
net.state_dict()['tag_classifier.conv6_4.bias'][0:] = torch.from_numpy(tag_premodel['conv6_4/b']).float()
net.state_dict()['encode1.weight'][0:] = torch.from_numpy(feature_premodel['encode1/W']).float()
net.state_dict()['encode1.bias'][0:] = torch.from_numpy(feature_premodel['encode1/b']).float()
net.state_dict()['encode2.weight'][0:] = torch.from_numpy(feature_premodel['encode2/W']).float()
net.state_dict()['encode2.bias'][0:] = torch.from_numpy(feature_premodel['encode2/b']).float()

for p in net.parameters():
    p.requires_grad = True

torch.save(net.state_dict(), 'models/i2v_pretrained_ver200_pytorch.bin')
