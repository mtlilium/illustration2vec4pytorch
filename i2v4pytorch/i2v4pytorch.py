from i2v4pytorch.base import Illustration2VecBase
from i2v4pytorch.i2v_net import I2VNet
import os
import numpy as np
import json
import torch
import torchvision.transforms as T
import torch.nn.functional as F
from torch.autograd import Variable

from scipy.ndimage import zoom
from skimage.transform import resize


class PytorchI2V(Illustration2VecBase):
    def __init__(self, *args, **kwargs):
        super(PytorchI2V, self).__init__(*args, **kwargs)
        self.caffe_mean = np.array([ 164.76139251,  167.47864617,  181.13838569])
        # we use pytorch mean and std on imagenet
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])

        self._init_param()

    def resize_image(self, im, new_dims, interp_order=1):
        # NOTE: we import the following codes from caffe.io.resize_image()
        if im.shape[-1] == 1 or im.shape[-1] == 3:
            im_min, im_max = im.min(), im.max()
            if im_max > im_min:
                # skimage is fast but only understands {1,3} channel images
                # in [0, 1].
                im_std = (im - im_min) / (im_max - im_min)
                resized_std = resize(im_std, new_dims, order=interp_order, mode='constant')
                resized_im = resized_std * (im_max - im_min) + im_min
            else:
                # the image is a constant -- avoid divide by 0
                ret = np.empty((new_dims[0], new_dims[1], im.shape[-1]),
                               dtype=np.float32)
                ret.fill(im_min)
                return ret
        else:
            # ndimage interpolates anything but more slowly.
            scale = tuple(np.array(new_dims) / np.array(im.shape[:2]))
            resized_im = zoom(im, scale + (1,), order=interp_order)
        return resized_im.astype(np.float32)

    def _image_loader(self, image_name):
        # image size should be 224x224(resized after PIL opening)
        loader = T.Compose([
                            #T.Resize(224),
                            T.ToTensor(),
                            T.Normalize(mean = self.mean,
                                        std = self.std)])
        for img in image_name: # batch
            img = img[:, :, [2, 1, 0]] # RGB -> BGR
            image_name = loader(img).unsqueeze(0)
        return image_name

    def _extract(self, inputs):
        shape = [len(inputs), 224, 224, 3]
        input_ = np.zeros(shape, dtype=np.float32)
        for ix, in_ in enumerate(inputs):
            input_[ix] = self.resize_image(in_, shape[1:])
        input_ = input_[:, :, :, ::-1]  # RGB to BGR
        input_ -= self.caffe_mean  # subtract mean
        input_ = input_.transpose((0, 3, 1, 2))  # (N, H, W, C) -> (N, C, H, W)
        input_ = torch.from_numpy(input_.copy()).float()
        inputs = self._image_loader(inputs)
        inputs = Variable(input_).to(self.device)
        return self.net(inputs)

    def _init_param(self):
        load_weights = torch.load(self.param_path, map_location={'cuda:0': 'cpu'})
        self.net.load_state_dict(load_weights, strict=False)
        del load_weights

'''
##########################################
'''

def make_i2v_with_pytorch(param_path=None,
                          tag_path=None,
                          threshold_path=None,
                          mode='tag'):

    kwargs = {}

    if param_path is not None:
        kwargs['param_path'] = param_path

    if tag_path is not None:
        tags = json.loads(open(tag_path, 'r').read())
        assert(len(tags) == 1539)
        kwargs['tags'] = tags

    if threshold_path is not None:
        fscore_threshold = np.load(threshold_path)['threshold']
        kwargs['threshold'] = fscore_threshold

    kwargs['mode'] = mode
    kwargs['device'] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net = I2VNet(mode=mode, tag_len=len(tags))
    return PytorchI2V(net, **kwargs)
