# Importing necessary libraries
import cv2
import torch
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
from utils.tensorboard import TensorBoard
from Renderer.model import FCN
from Renderer.stroke_gen import *

# Creating TensorBoard writer log for training progress
writer = TensorBoard("../train_log/")

# Optimizer class
import torch.optim as optim

# Mean squared error loss function
criterion = nn.MSELoss()

net = FCN()
# Adam optimizer
optimizer = optim.Adam(net.parameters(), lr=3e-6)
batch_size = 64

# Check of CUDA is available
use_cuda = torch.cuda.is_available()

# initializing step counter
step = 0

# Saves model paramenters
def save_model():
    # if CUDA available, move model to CPU
    if use_cuda:
        net.cpu()
    # Saving model parameters to a file
    torch.save(net.state_dict(), "C:\\Users\\Garima Ranjan\\Downloads\\Git\\RL project\\ICCV2019-LearningToPaint\\baseline\\DRL\\renderer.pkl")
    # if CUDA available, move model to GPU
    if use_cuda:
        net.cuda()

# For updating model parameters(weights)
def load_weights():
    # loading parameters from file
    pretrained_dict = torch.load("C:\\Users\\Garima Ranjan\\Downloads\\Git\\RL project\\ICCV2019-LearningToPaint\\baseline\\DRL\\renderer.pkl")

    # getting current state of model parameters
    model_dict = net.state_dict()

    # Filter parameters, to include only the ones present in current model
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

    # update current model parameters with pretrained model parameters
    model_dict.update(pretrained_dict)

    # load the updated parameters
    net.load_state_dict(model_dict)


load_weights()

# Starting training loop
# For better training we can use while step < 500000:
while step < 5000:
    net.train()
    train_batch = []
    ground_truth = []
    for i in range(batch_size):
        f = np.random.uniform(0, 1, 10)
        train_batch.append(f)
        ground_truth.append(draw(f))

    train_batch = torch.tensor(train_batch).float()
    ground_truth = torch.tensor(ground_truth).float()
    if use_cuda:
        net = net.cuda()
        train_batch = train_batch.cuda()
        ground_truth = ground_truth.cuda()
    gen = net(train_batch)
    optimizer.zero_grad()
    loss = criterion(gen, ground_truth)
    loss.backward()
    optimizer.step()
    print(step, loss.item())
    if step < 200000:
        lr = 1e-4
    elif step < 400000:
        lr = 1e-5
    else:
        lr = 1e-6
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    writer.add_scalar("train/loss", loss.item(), step)
    if step % 10 == 0:
        net.eval()
        gen = net(train_batch)
        loss = criterion(gen, ground_truth)
        writer.add_scalar("val/loss", loss.item(), step)
        for i in range(32):
            G = gen[i].cpu().data.numpy()
            GT = ground_truth[i].cpu().data.numpy()
            writer.add_image("train/gen{}.png".format(i), G, step)
            writer.add_image("train/ground_truth{}.png".format(i), GT, step)
    if step % 10 == 0:
        save_model()
    step += 1
