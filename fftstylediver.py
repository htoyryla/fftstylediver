# @htoyryla Hannu Töyrylä 27 Dec 2021

# FFT + VGG19 style transfer + CLIP 


import torch
import torch.nn.functional as F
from torchvision.utils import save_image
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
import torchvision.models as models
from PIL import Image
import argparse
import imageio
import numpy as np
import os
import clip
import kornia.geometry as geometry

from cutouts import cut

import yaml
import torch.nn as nn


# use command line parameters
parser = argparse.ArgumentParser()

# define params and their types with defaults if needed

parser.add_argument('--image', type=str, default="test.png", help='path to image')
parser.add_argument('--style', type=str, default="", help='path to style image')
parser.add_argument('--lr', type=float, default=0.05, help='learning rate')
parser.add_argument('--niter', type=int, default=1000, help='number of iterations')
parser.add_argument('--name', type=str, default="out/harj3", help='basename for storing images')
parser.add_argument('--show', action="store_true", help='show image in a window')
parser.add_argument('--imageSize', type=int, default=256, help='image size')
parser.add_argument('--cuda', action="store_true", help='use cuda if available')
parser.add_argument('--saveEvery', type=int, default=10, help='image save frequency')
parser.add_argument('--cw', type=float, default=10, help='content weight')
parser.add_argument('--sw', type=float, default=10, help='style weight')

parser.add_argument('--init', type=str, default="random", help='random | path to image')
parser.add_argument('--content_layer', type=int, default=10, help='content layer index')
parser.add_argument('--style_layers', type=int, nargs='*', default=[1, 3, 5, 9, 13, 15], help='style layers indices')
parser.add_argument('--style_scale', type=float, default=1, help='')
parser.add_argument('--content_scale', type=float, default=1, help='')
parser.add_argument('--text', type=str, default="", help='text prompt')

parser.add_argument('--showSize', type=int, default=0, help='image size for onscreen display')

parser.add_argument('--tw', type=float, default=2, help='text weight')
parser.add_argument('--low', type=float, default=0.4, help='lower limit for cut scale')
parser.add_argument('--high', type=float, default=1.0, help='higher limit for cut scale')
parser.add_argument('--cutn', type=int, default=8, help='number of cutouts for CLIP')

parser.add_argument('--decay', type=float, default=0.93, help='decay factor')
parser.add_argument('--sd', type=float, default=0.001, help='params init scale')
parser.add_argument('--colors', type=float, default=1.5, help='colors factor')
parser.add_argument('--fcontrast', type=float, default=1, help='contrast factor')
parser.add_argument('--initmult', type=float, default=0.1, help='init image to paras scale factor')
parser.add_argument('--tvw', type=float, default=2, help='tv weight') # NOT USED


# get params into opt object, so we can access them like opt.image

opt = parser.parse_args()

# import opencv image processing library and numpy


import cv2
import numpy

# NOT USED

class TVLoss(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(TVLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,1:])
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]
        
tvloss = TVLoss(TVLoss_weight=opt.tvw)   


### FFT image gen

# based on https://github.com/eps696/aphantasia

'''
def pixel_image(shape, sd=2.):
        tensor = (torch.randn(*shape) * sd).cuda().requires_grad_(True)
        return [tensor], lambda: tensor
'''

# From https://github.com/tensorflow/lucid/blob/master/lucid/optvis/param/spatial.py

def rfft2d_freqs(h, w):
        """Computes 2D spectrum frequencies."""
        fy = np.fft.fftfreq(h)[:, None]
        fx = np.fft.fftfreq(w) 
        return np.sqrt(fx * fx + fy * fy)

def resume_fft(resume=None, shape=None, decay=None, colors=1.6, sd=0.01):
    size = (opt.imageSize, opt.imageSize)
    if resume is None: # random init
        params_shape = [*shape[:4], 2] 
        print(params_shape)
        params = sd * torch.randn(*params_shape).cuda()
    elif isinstance(resume, str):
        if os.path.isfile(resume):
            if os.path.splitext(resume)[1].lower()[1:] in ['jpg','png','tif','bmp']:
                img_in = imageio.imread(resume).astype(np.float32)
                print(img_in.shape)
                params = img2fft(img_in, decay, colors)
            else:
                params = torch.load(resume)
                if isinstance(params, list): params = params[0]
                params = params.detach().cuda()
            params *= sd
        else: print(' Snapshot not found:', resume); exit()
    else:
        if isinstance(resume, list): resume = resume[0]
        params = resume.cuda()
    return params, size

def fft_image(shape, sd=0.01, decay_power=1.0, resume=None): # decay ~ blur

    params, size = resume_fft(resume, shape, decay_power, sd=sd)
    spectrum_real_imag_t = params.requires_grad_(True)
    if size is not None: shape[2:] = size
    [h,w] = list(shape[2:])

    freqs = rfft2d_freqs(h,w)
    scale = 1. / np.maximum(freqs, 4./max(h,w)) ** decay_power
    scale *= np.sqrt(h*w)
    scale /= scale.max()
    scale = torch.tensor(scale).float()[None, None, ..., None].cuda()
    print(scale.shape)

    def inner(shift=None, contrast=1.):
        scaled_spectrum_t = scale * spectrum_real_imag_t
        if shift is not None:
           scaled_spectrum_t += scale * shift
        if float(torch.__version__[:3]) < 1.8:
            image = torch.irfft(scaled_spectrum_t, 2, normalized=True, onesided=False)
        else:
            if type(scaled_spectrum_t) is not torch.complex64:
                scaled_spectrum_t = torch.view_as_complex(scaled_spectrum_t)
            image = torch.fft.irfftn(scaled_spectrum_t, s=(h, w), norm='ortho')
        image = image * contrast / image.std() # keep contrast, empirical
        return image

    return [spectrum_real_imag_t], inner, size

def inv_sigmoid(x):
    eps = 1.e-12
    x = torch.clamp(x.double(), eps, 1-eps)
    y = torch.log(x/(1-x))
    return y.float()

def un_rgb(image, colors=1.):
    color_correlation_svd_sqrt = np.asarray([[0.26, 0.09, 0.02], [0.27, 0.00, -0.05], [0.27, -0.09, 0.03]]).astype("float32")
    color_correlation_svd_sqrt /= np.asarray([colors, 1., 1.])
    max_norm_svd_sqrt = np.max(np.linalg.norm(color_correlation_svd_sqrt, axis=0))
    color_correlation_normalized = color_correlation_svd_sqrt / max_norm_svd_sqrt
    color_uncorrelate = np.linalg.inv(color_correlation_normalized)

    image = inv_sigmoid(image)
    t_permute = image.permute(0,2,3,1)
    t_permute = torch.matmul(t_permute, torch.tensor(color_uncorrelate.T).cuda())
    image = t_permute.permute(0,3,1,2)
    return image

def un_spectrum(spectrum, decay_power):
    h = spectrum.shape[2]
    w = spectrum.shape[3]
    freqs = rfft2d_freqs(h, w)
    scale = 1.0 / np.maximum(freqs, 1.0 / max(w, h)) ** decay_power
    scale *= np.sqrt(w*h)
    scale = torch.tensor(scale).float()[None, None, ..., None].cuda()
    scale = scale / scale.max()
    return spectrum / scale

def img2fft(img_in, decay=1., colors=1.):
    img_in = torch.Tensor(img_in).cuda().permute(2,0,1).unsqueeze(0) / 255.
    img_in = geometry.transform.resize(img_in, (opt.imageSize, opt.imageSize))
    h, w = img_in.shape[0], img_in.shape[1]
    
    with torch.no_grad():
        if float(torch.__version__[:3]) < 1.8:
            spectrum = torch.rfft(img_in, 2, normalized=True, onesided=False) # 1.7
        else:
            spectrum = torch.fft.rfftn(img_in, s=(h, w), dim=[2,3], norm='ortho') # 1.8
            spectrum = torch.view_as_real(spectrum)
        spectrum = un_spectrum(spectrum, decay_power=decay)
        spectrum *= opt.initmult / opt.sd 
    return spectrum

'''
def to_valid_rgb(image_f, colors=1., decorrelate=True):
    color_correlation_svd_sqrt = np.asarray([[0.26, 0.09, 0.02],
                                             [0.27, 0.00, -0.05],
                                             [0.27, -0.09, 0.03]]).astype("float32")
    color_correlation_svd_sqrt /= np.asarray([colors, 1., 1.]) # saturate, empirical
    max_norm_svd_sqrt = np.max(np.linalg.norm(color_correlation_svd_sqrt, axis=0))
    color_correlation_normalized = color_correlation_svd_sqrt / max_norm_svd_sqrt

    def _linear_decorrelate_color(tensor):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        t_permute = tensor.permute(0,2,3,1)
        t_permute = torch.matmul(t_permute, torch.tensor(color_correlation_normalized.T).to(device))
        tensor = t_permute.permute(0,3,1,2)
        return tensor

    def inner(*args, **kwargs):
        image = image_f(*args, **kwargs)
        if decorrelate:
            image = _linear_decorrelate_color(image)
        return torch.sigmoid(image)
    return inner
'''

# part based on aphantasia ends here

# define display routine

def show_on_screen(image_tensor, window="out", size=0):
    im = image_tensor.detach().numpy()   # convert from pytorch tensor to numpy array
    
    # pytorch tensors are (C, H, W), rearrange to (H, W, C)
    im = im.transpose(1, 2, 0)
    
    # adjust range to 0 .. 1
    im -= im.min()
    im /= im.max()

    if size > 0:
        im = cv2.resize(im, (size, size))

    # show it in a window (this will not work on a remote session)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)    
    cv2.imshow(window, im)
    cv2.waitKey(100)   # display for 100 ms and wait for a keypress (which we ignore here)

cuda = opt.cuda

if cuda:
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
  device = torch.device("cpu")
  
# LOAD CLIP

perceptor, clip_preprocess = clip.load('ViT-B/32', jit=False)
perceptor = perceptor.eval()

cnorm = Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
  
# next we need the text embedding for the prompt

tx = clip.tokenize(opt.text)                        # convert text to a list of tokens 
txt_enc = perceptor.encode_text(tx.to(device)).detach()   # get sentence embedding for the tokens
del tx

# first make a list of transforms needed in correct order

xforms = []
xforms.append(Resize(opt.imageSize))    # resize image
xforms.append(ToTensor())            # convert to pytorch tensor
xforms.append(Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)))            # normalize to range -1...1

# compose into a transform pipeline

preprocess = Compose(xforms)

xsforms = []
stylesize = int(opt.style_scale * opt.imageSize)
xsforms.append(Resize(stylesize))    # resize image
xsforms.append(ToTensor())            # convert to pytorch tensor
xsforms.append(Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)))            # normalize to range -1...1

# compose into a transform pipeline

style_preprocess = Compose(xsforms)

# read and  preprocess target and style images

filename = opt.image 
imgC = preprocess(Image.open(opt.image).convert("RGB")).to(device) #.clamp_(-2,2)

filename = opt.style #"test.png"
imgS = style_preprocess(Image.open(filename).convert("RGB")).to(device) #.clamp_(-2,2)

# prepare initial image

if opt.init == "random":
    imgG = torch.zeros_like(imgC).uniform_(-1,1).to(device)  # NOT USED  
else:
    # initialize from a given image (other than content img)
    imgG = preprocess(Image.open(opt.init).convert("RGB")).to(device) #.clamp_(-2,2)    


#norm = Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))



lr = opt.lr 

# let us first create a VGG19 network

vgg = models.vgg19(pretrained=True).features.to(device).eval()

# list of suitable layers for content/style evaluation

players = [1,3,6,8,11,13,15,17,20,22,24,26,29,31,33,35]

content_idx = opt.content_layer
style_idxs = opt.style_layers #[1, 3, 5, 9, 13, 15] 

content_layer = vgg[players[content_idx]]

print(content_layer) 

# add a hook to read output of content layer

content_acts = [None]
def content_hook(i):
    def hook(model, input, output):
        content_acts[i] = output
    return hook
        
content_layer.register_forward_hook(content_hook(0))

# add a hook to read style evaluation from selected layers

# use a gram matrix to evaluate style (texture) instead of content

def gram(input):
    a, b, c, d = input.size() 
    f = input.clone().reshape(a * b, c * d)  # resise F_XL into \hat F_XL
    gr = torch.mm(f, f.t())  # compute the gram product
    return gr.div(a * b * c * d)
    

# now add the actual style hooks    

style_grams = [None]*len(style_idxs)
def style_hook(i):
    def hook(model, input, output):
        style_grams[i] = gram(output)
    return hook

n = 0
for s in style_idxs:
   style_layer = vgg[players[s]]    
   style_layer.register_forward_hook(style_hook(n))
   n += 1
     
# now we need targets for content and style     
     
# feed content image to VGG and store output from content hook
     
o = vgg(imgC.unsqueeze(0))
content_targets = content_acts[0].detach() #.shape  

# feed style image to VGG and store outputs from style hooks

style_targets = [None]*len(style_idxs)
o = vgg(imgS.unsqueeze(0))
for n in range(len(style_idxs)):      
    style_targets[n] = style_grams[n].detach() #.shape        

# get ready to iterate
        
niter = opt.niter

resume = None
if opt.init !="random" and opt.init != "image":
   resume = opt.init

shape = [1, 3, opt.imageSize, opt.imageSize]
params, image_f, sz = fft_image(shape, sd=opt.sd, decay_power=opt.decay, resume=resume)

#image_f = to_valid_rgb(image_f, colors = opt.colors)
    
optimizer = torch.optim.Adam(params, opt.lr)


run = 0

# then go

while run <=  niter:
    
    optimizer.zero_grad()
    
    # get current image and feed into VGG

    imgO = image_f(contrast = opt.fcontrast)
    imgO_ = imgO.clone() 

    o = vgg(imgO_)
    
    # store content and style actuals
    
    content_actuals = content_acts[0]
    
    style_actuals = [None]*len(style_idxs)
    
    # evaluate content loss
    
    lossc = opt.cw * F.mse_loss(content_targets, content_actuals)
 
    # evaluate total style loss
 
    style_losses = []
    losss = 0
    for n in range(len(style_idxs)):      
        style_actuals[n] = style_grams[n]
        sl = opt.sw * F.mse_loss(style_targets[n], style_actuals[n])
        style_losses.append(sl.item())
        losss += sl
     
    # text loss
    
    # prepare image for CLIP by making random cutouts
  
    nimg = (imgO_/2 + 1) / 2     
    nimg = cut(nimg, cutn=opt.cutn, low=opt.low, high=opt.high, norm = cnorm)
 
    # get image encoding from CLIP
 
    img_enc = perceptor.encode_image(nimg) 
  
    # we already have text embedding for the promt in txt_enc
    # so we can evaluate similarity
  
    losst = opt.tw*(1-torch.cosine_similarity(txt_enc, img_enc)).view(-1, 1).T.mean(1)
    
    
    loss = lossc + losss + losst  

    loss.backward()
    
    # print loss to show how we are doing

    print(run, lossc.item(), losss.item(), losst.item(), loss.item())
 
    # save image
    if run % opt.saveEvery == 0:
          save_image(imgO_.clone().detach(), opt.name+"-"+str(run)+".jpg", normalize=True)    

    # show on screen
    if opt.show:
          show_on_screen(imgO_[0].cpu(), opt.name, size=opt.showSize)# print loss to show how we are doing
  
    run += 1
    
    optimizer.step()
  
    
