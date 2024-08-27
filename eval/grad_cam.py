import os, sys, glob
sys.path.insert(0, os.path.abspath('.'))
import network.ael_net as net
import cv2
import numpy as np
import torch
from tqdm import tqdm
from network import load_model
from skimage.io import imread
from torch.nn import Module
import torch.nn as nn
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from torchcam.methods import SmoothGradCAMpp
from torchvision.transforms.functional import normalize, to_pil_image, to_tensor#, resize
from PIL import Image
from torchvision import datasets, transforms
from torchcam.utils import overlay_mask
from data import data_loader

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') #

class GradCamModel(Module):
    def __init__(self, model, eval_layer):
        super().__init__()
        self.gradients = None
        self.tensorhook = []
        self.layerhook = []
        self.selected_out = None
        self.model = model

        self.layerhook.append(eval_layer.register_forward_hook(self.forward_hook()))
        
        for p in self.model.parameters():
            p.requires_grad = True
    
    def activations_hook(self, grad):
        self.gradients = grad

    def get_act_grads(self):
        return self.gradients

    def forward_hook(self):
        def hook(module, inp, out):
            self.selected_out = out
            self.tensorhook.append(out.register_hook(self.activations_hook))
        return hook

    def forward(self, x):
        out = self.model(x)
        return out, self.selected_out

def gradcam_cbam(gradcam_model, eval_layer, image_path, root_dir='./graphs/heatmap', method='', modal='peri', base_img_pixels=(112, 112), heatmap_style='jet', aleph=0.9, base_image=True, figsize=(5,5)):
    from torchvision.transforms.functional import resize
    gradcam_model = SmoothGradCAMpp(model, target_layer = eval_layer, input_shape=(3, 112, 112))
    
    new_img_name = os.path.join(root_dir, method, modal)
    transform = transforms.Compose([    transforms.Normalize(
                                    mean=[-0.5/0.5, -0.5/0.5, -0.5/0.5],
                                    std=[1/0.5, 1/0.5, 1/0.5]),
                                    transforms.ToPILImage()
                               ])
    
    data_load, data_set = data_loader.gen_data(image_path, 'test', type=modal, aug='False')

    for i, (img_in) in tqdm(enumerate(data_set)):
        img_path = data_set.imgs[i][0]
        img_name = img_path.split('/')[-1].split('.')[0]
        pil_img = resize(Image.open(img_path), (112, 112))
        img_tensor = normalize(to_tensor(pil_img), [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

        scores = model(img_tensor.unsqueeze(0).to(device))
        scores = scores[0]
        class_idx = scores.argmax().item()
        activation_map = gradcam_model(img_in[1], scores)
        activation_map = [activation_map[0].detach().cpu()]
        if modal == 'peri':
            base_img_pixels = (37, 112)
        result = overlay_mask(resize(pil_img, base_img_pixels), resize(to_pil_image(activation_map[0].squeeze(0), mode='F'), base_img_pixels), alpha=aleph)

        path = os.path.join(new_img_name, (img_path.split('/')[-2] + '_' + img_name))
        resize(pil_img, base_img_pixels).save(path + '_ori.png', 'PNG')
        result.save(path + '_c.png', 'PNG')


def gradcam_img(model, eval_layer, image_path, root_dir='./graphs/heatmap', method='', modal='peri', base_img_pixels=(112, 112), heatmap_style='jet', aleph=0.9, base_image=True, figsize=(5,5)):
    from skimage.transform import resize
    gradcam_model = GradCamModel(model, eval_layer).to(device)
    
    # some parameters
    new_img_name = os.path.join(root_dir, method, modal)
    cmap = mpl.cm.get_cmap(heatmap_style, 256)

    # read image and normalize
    img = imread(image_path)
    img_name = image_path.split('/')[-1].split('.')[0]
    new_img_name = os.path.join(root_dir, method, modal)
    img = resize(img, base_img_pixels, preserve_range = True)
    img = np.expand_dims(img.transpose((2, 0, 1)), 0)
    img /= 255.0
    mean = np.array([0.5, 0.5, 0.5]).reshape((1, 3, 1, 1))
    std = np.array([0.5, 0.5, 0.5]).reshape((1, 3, 1, 1))
    img = (img - mean) / std
    inp_img = torch.from_numpy(img).to(device, torch.float32)

    # get feature and backpropagate
    out, acts = gradcam_model(inp_img)
    acts = acts.detach().cpu()
    loss = nn.CrossEntropyLoss()(out, torch.from_numpy(np.array([0])).to(device))
    loss.backward()
    grads = gradcam_model.get_act_grads().detach().cpu()
    pooled_grads = torch.mean(grads, dim=[0, 2, 3]).detach().cpu()

    if modal == 'peri' or modal == 'periocular':
        base_img_pixels = (37, 112)

    # plot channel-wise and mean gradcam images
    for i in range(acts.shape[1]):
        acts[:, i, :, :] += pooled_grads[i]
        heatmap_j = torch.mean(acts[:, i, :, :], dim = 0)        
        heatmap_j = resize(heatmap_j, base_img_pixels, preserve_range=True)
        heatmap_j2 = cmap(heatmap_j, alpha = aleph)
        fig, axs = plt.subplots(1, 1, figsize = figsize)
        axs.xaxis.set_visible(False)
        axs.yaxis.set_visible(False) 
        if base_image == True:
            axs.imshow(resize((img * std + mean)[0].transpose(1, 2, 0), base_img_pixels, preserve_range = True))
        axs.imshow(heatmap_j2)
        # plt.savefig((new_img_name + '/channels/' + img_name + '_c' + str(i) + '.png'), bbox_inches='tight')
        plt.close(fig)

    heatmap_j = torch.mean(acts, dim = 1).squeeze()        
    heatmap_j_max = heatmap_j.max(axis = 0)[0]
    heatmap_j /= heatmap_j_max
    heatmap_j = resize(heatmap_j, base_img_pixels, preserve_range=True)
    heatmap_j2 = cmap(heatmap_j, alpha = aleph)
    fig, axs = plt.subplots(1, 1)
    axs.xaxis.set_visible(False)
    axs.yaxis.set_visible(False)
    if base_image == True:
        axs.imshow(resize((img * std + mean)[0].transpose(1, 2, 0), base_img_pixels, preserve_range = True), alpha=1.0)
    axs.imshow(heatmap_j2)    
    plt.savefig((new_img_name + '/mean/' + img_name + '_mean.png'), bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    '''
        root dir: main directory to store all images
        method: subfolder for method
        modal: periocular or face
        pixel_size: size of image in pixels
        heatmap_style: style of heatmap used
        aleph: opacity of the heatmap shown, 1 = least opaque, 0 = most opaque
        base_image: True to show the original image with the heatmap, else False to just show heatmap
        figsize: size of figure in matplotlib
        image_path: change image path, create a loop if to be used with multiple images
    '''

    root_dir = './graphs/heatmap'
    model_layer = 'AELNet'
    method = 'AELNet'
    modal = ['face','peri']
    pixel_size = (112, 112)
    heatmap_style = 'jet'
    aleph = 0.5
    base_image = True
    figsize = (5, 5)

    for modality in modal:
        modality = modality[:4]
        if not os.path.exists(os.path.join(root_dir, str(model_layer + method), modality)):
            os.makedirs(os.path.join(root_dir, str(model_layer + method), modality, 'channels'))
            os.makedirs(os.path.join(root_dir, str(model_layer + method), modality, 'mean'))
    
    # load model and set evaluation layer for GradCAM
    embd_dim = 1024
    model = net.AEL_Net(embedding_size=embd_dim).eval()    
    load_model_path = '/home/tiongsik/Python/ael_net/models/best_model/AELNet.pth'
    model = load_model.load_pretrained_network(model, load_model_path, device=device).eval()
    eval_layer = model.conv_6_sep
            
    # # plot multiple images
    for modality in modal:
        image_path = '/home/tiongsik/Python/conditional_biometrics/data/gradcam_imgs/' + str(modality) + '/1/'
        gradcam_cbam(model, eval_layer, image_path, root_dir=root_dir,
                method=str(model_layer + method), modal=modality, base_img_pixels=pixel_size, 
                heatmap_style=heatmap_style, aleph=(1-aleph), base_image=base_image, figsize=figsize)
            