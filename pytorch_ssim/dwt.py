import pywt
import torch
import torch.nn.functional as F
from torch.autograd import Variable

w = pywt.Wavelet('db1')

dec_hi = torch.Tensor(w.dec_hi[::-1])
dec_lo = torch.Tensor(w.dec_lo[::-1])
rec_hi = torch.Tensor(w.rec_hi)
rec_lo = torch.Tensor(w.rec_lo)

Lfilters = torch.stack([dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1)], dim=0)
Mfilters = torch.stack([
    dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),
    dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1)
],
                       dim=0)
Hfilters = torch.stack([dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)], dim=0)


def dwt(img):
    Lfilters_cat = torch.cat(tuple(Lfilters[:, None]) * img.shape[1], 0)
    Lfilters_cat = Lfilters_cat.unsqueeze(1)
    Mfilters_cat = torch.cat(tuple(Mfilters[:, None]) * img.shape[1], 0)
    Mfilters_cat = Mfilters_cat.unsqueeze(1)
    Hfilters_cat = torch.cat(tuple(Hfilters[:, None]) * img.shape[1], 0)
    Hfilters_cat = Hfilters_cat.unsqueeze(1)
    return F.conv2d(img, Variable(Lfilters_cat.cuda(),requires_grad=True),stride=2, groups=img.shape[1]) \
        ,F.conv2d(img, Variable(Mfilters_cat.cuda(),requires_grad=True),stride=2, groups=img.shape[1]) \
        ,F.conv2d(img, Variable(Hfilters_cat.cuda(),requires_grad=True),stride=2, groups=img.shape[1])
