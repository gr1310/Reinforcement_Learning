# Importing necessary libraries
import os
import cv2
import torch
import numpy as np
import argparse
import torch.nn as nn
import torch.nn.functional as F

# Imports RL agent
from DRL.actor import *

# Imports Renderer
from Renderer.stroke_gen import *

# Imports model classes
from Renderer.model import *

# Setting device for training (cpu/gpu)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Setting width of output image
width = 128

# Creating an argument parser
parser = argparse.ArgumentParser(description='Learning to Paint')
# adding max_step, actor, renderer, img, imgid, and divide arguments to the parser
parser.add_argument('--max_step', default=40, type=int, help='max length for episode')
parser.add_argument('--actor', default="C:\\Users\\Garima Ranjan\\Downloads\\Git\\RL project\\ICCV2019-LearningToPaint\\baseline\\DRL\\actor.pkl", type=str, help='Actor model')
parser.add_argument('--renderer', default="C:\\Users\\Garima Ranjan\\Downloads\\Git\\RL project\\ICCV2019-LearningToPaint\\baseline\\DRL\\renderer.pkl", type=str, help='renderer model')
parser.add_argument('--img', default='image/test.png', type=str, help='test image')
parser.add_argument('--imgid', default=0, type=int, help='set begin number for generated image')
parser.add_argument('--divide', default=4, type=int, help='divide the target image to get better resolution')
# Parsing the arguments
args = parser.parse_args()

# calculating number of canvases
canvas_cnt = args.divide * args.divide

# Creating tensor of ones to represent time steps
T = torch.ones([1, 1, width, width], dtype=torch.float32).to(device)

# Reading the target image
img = cv2.imread(args.img, cv2.IMREAD_COLOR)
# Storing shape of target image
origin_shape = (img.shape[1], img.shape[0])

# Creating tensor of zeroes
# Will be used to represent coordinates of each pixel in canvas
coord = torch.zeros([1, 2, width, width])

for i in range(width):  #iteration through width of canvas
    for j in range(width):  #iteration through height of canvas
        coord[0, 0, i, j] = i / (width - 1.)  #normalized x-coordinate
        coord[0, 1, i, j] = j / (width - 1.)  #normalized y-coordinate

# Moving the coordinates tensor to the device
coord = coord.to(device) # Coordconv

Decoder = FCN()
# loading pretrained decoder model
Decoder.load_state_dict(torch.load(args.renderer))

# Updates canvas
# x= action vector
def decode(x, canvas): # b * (10 + 3)
    # Reshaping action vector to be a tensor of shape(batch_size,13)
    x = x.view(-1, 10 + 3)
    """The first 10 elements of the action vector represent the parameters of the stroke, and the last 3 elements of the action vector represent the color of the stroke."""
    stroke = 1 - Decoder(x[:, :10]) 
    stroke = stroke.view(-1, width, width, 1)
    color_stroke = stroke * x[:, -3:].view(-1, 1, 1, 3)
    #generates stroke image from action vector
    stroke = stroke.permute(0, 3, 1, 2)
    color_stroke = color_stroke.permute(0, 3, 1, 2)
    stroke = stroke.view(-1, 5, 1, width, width)
    color_stroke = color_stroke.view(-1, 5, 3, width, width)
    res = []
    for i in range(5):
        # updates the canvas using the stroke image and the color of the stroke
        canvas = canvas * (1 - stroke[:, i]) + color_stroke[:, i]
        res.append(canvas)

    # returns updated canvas and a list of intermediate canvases
    return canvas, res

def small2large(x):
    # (d * d, width, width) -> (d * width, d * width)    
    x = x.reshape(args.divide, args.divide, width, width, -1)
    x = np.transpose(x, (0, 2, 1, 3, 4))
    x = x.reshape(args.divide * width, args.divide * width, -1)
    return x

def large2small(x):
    # (d * width, d * width) -> (d * d, width, width)
    x = x.reshape(args.divide, width, args.divide, width, 3)
    x = np.transpose(x, (0, 2, 1, 3, 4))
    x = x.reshape(canvas_cnt, width, width, 3)
    return x

def smooth(img):
    def smooth_pix(img, tx, ty):
        if tx == args.divide * width - 1 or ty == args.divide * width - 1 or tx == 0 or ty == 0: 
            return img
        img[tx, ty] = (img[tx, ty] + img[tx + 1, ty] + img[tx, ty + 1] + img[tx - 1, ty] + img[tx, ty - 1] + img[tx + 1, ty - 1] + img[tx - 1, ty + 1] + img[tx - 1, ty - 1] + img[tx + 1, ty + 1]) / 9
        return img

    for p in range(args.divide):
        for q in range(args.divide):
            x = p * width
            y = q * width
            for k in range(width):
                img = smooth_pix(img, x + k, y + width - 1)
                if q != args.divide - 1:
                    img = smooth_pix(img, x + k, y + width)
            for k in range(width):
                img = smooth_pix(img, x + width - 1, y + k)
                if p != args.divide - 1:
                    img = smooth_pix(img, x + width, y + k)
    return img

def save_img(res, imgid, divide=False):
    output = res.detach().cpu().numpy() # d * d, 3, width, width    
    output = np.transpose(output, (0, 2, 3, 1))
    if divide:
        output = small2large(output)
        output = smooth(output)
    else:
        output = output[0]
    output = (output * 255).astype('uint8')
    output = cv2.resize(output, origin_shape)
    cv2.imwrite('output/generated' + str(imgid) + '.png', output)

# Loads the pretrained actor model
actor = ResNet(9, 18, 65) # action_bundle = 5, 65 = 5 * 13
actor.load_state_dict(torch.load(args.actor))
actor = actor.to(device).eval()
Decoder = Decoder.to(device).eval()

# initializing canvas
canvas = torch.zeros([1, 3, width, width]).to(device)

# resizing target image to match canvas size
patch_img = cv2.resize(img, (width * args.divide, width * args.divide))
patch_img = large2small(patch_img)
patch_img = np.transpose(patch_img, (0, 3, 1, 2))
patch_img = torch.tensor(patch_img).to(device).float() / 255.

img = cv2.resize(img, (width, width))
img = img.reshape(1, width, width, 3)
img = np.transpose(img, (0, 3, 1, 2))
img = torch.tensor(img).to(device).float() / 255.

# Directory for storing output images
os.system('mkdir output')

# Start generating images
with torch.no_grad():
    if args.divide != 1:
        args.max_step = args.max_step // 2
    for i in range(args.max_step):
        # Calculating time step
        stepnum = T * i / args.max_step

        # Action from actor model
        actions = actor(torch.cat([canvas, img, stepnum, coord], 1))

        # Updating canvas
        canvas, res = decode(actions, canvas)

        # printing loss
        print('canvas step {}, L2Loss = {}'.format(i, ((canvas - img) ** 2).mean()))

        # saving intermediate canvases
        for j in range(5):
            save_img(res[j], args.imgid)
            args.imgid += 1

    if args.divide != 1:
        # Saving final image
        canvas = canvas[0].detach().cpu().numpy()
        canvas = np.transpose(canvas, (1, 2, 0))    
        canvas = cv2.resize(canvas, (width * args.divide, width * args.divide))
        canvas = large2small(canvas)
        canvas = np.transpose(canvas, (0, 3, 1, 2))
        canvas = torch.tensor(canvas).to(device).float()
        coord = coord.expand(canvas_cnt, 2, width, width)
        T = T.expand(canvas_cnt, 1, width, width)
        for i in range(args.max_step):
            stepnum = T * i / args.max_step
            actions = actor(torch.cat([canvas, patch_img, stepnum, coord], 1))
            canvas, res = decode(actions, canvas)
            print('divided canvas step {}, L2Loss = {}'.format(i, ((canvas - patch_img) ** 2).mean()))
            for j in range(5):
                save_img(res[j], args.imgid, True)
                args.imgid += 1
