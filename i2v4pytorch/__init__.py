from i2v4pytorch.base import Illustration2VecBase

pytorch_available = False

try:
    from i2v4pytorch.i2v4pytorch import PytorchI2V, make_i2v_with_pytorch
    pytorch_available = True
except ImportError:
    pass


if not pytorch_available:
    raise ImportError('i2v4pytorch requires pytorch package')
