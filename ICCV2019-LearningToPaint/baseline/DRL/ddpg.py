import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD
from Renderer.model import *
from DRL.rpm import rpm
from DRL.actor import *
from DRL.critic import *
from DRL.wgan import *

from utils.util import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

coord = torch.zeros([1, 2, 128, 128])
for i in range(128):
    for j in range(128):
        coord[0, 0, i, j] = i / 127.
        coord[0, 1, i, j] = j / 127.
coord = coord.to(device)

criterion = nn.MSELoss()

'''
1) renderer.pkl has parameters(weights, biases) of FCN.
'''

Decoder = FCN()
# Use the correct file location on local machine
Decoder.load_state_dict(torch.load('C:\\Users\\Garima Ranjan\\Downloads\\Git\\RL project\\ICCV2019-LearningToPaint\\baseline\\DRL\\renderer.pkl'))

def decode(x, canvas): # b * (10 + 3)
    """The decode function takes the output tensor from the neural network model and 
    decodes it to generate the corresponding stroke pattern and color strokes. It then updates 
    the initial canvas tensor based on the decoded information."""
    x = x.view(-1, 10 + 3)  # reshaping to 2D
    # decoder returns inverted probabilities
    # stroke has actual probabilites(prob again got inverted)
    stroke = 1 - Decoder(x[:, :10]) # taking first 10 dimensions of x
 
    stroke = stroke.view(-1, 128, 128, 1)   #reshaping to 4D - to match dimensions of canvas
    # color information
    color_stroke = stroke * x[:, -3:].view(-1, 1, 1, 3) # taking last 3 dimensions of x for color info
    # here .view() requires that the total number of elements in the tensor remains constant, whereas .permute() does not change the total number of elements.
    # Transposing the stroke tensor to change the order of its dimensions, making it compatible with the subsequent operations.
    stroke = stroke.permute(0, 3, 1, 2)
    color_stroke = color_stroke.permute(0, 3, 1, 2) #matching with stroke prob
    # The additional dimension (5) represents the five possible strokes (or actions) that can be taken at each step.
    stroke = stroke.view(-1, 5, 1, 128, 128)    #reshaping to 5D
    color_stroke = color_stroke.view(-1, 5, 3, 128, 128)

    # This loop iterates over the five possible strokes (or actions). 
    # For each stroke, it updates the canvas tensor by masking out the existing canvas with the stroke probability mask and adding the corresponding color stroke.
    for i in range(5):
         # here (1 - stroke) mask out old values and color_stroke adds new values
        canvas = canvas * (1 - stroke[:, i]) + color_stroke[:, i]
    return canvas

# Taking transpose and multiplying, returns transpose of whole thing
def cal_trans(s, t):
    return (s.transpose(0, 3) * t).transpose(0, 3)
    
# Deep Deterministic Policy Gradient algorithm
class DDPG(object):
    def __init__(self, batch_size=64, env_batch=1, max_step=40, \
                 tau=0.001, discount=0.9, rmsize=800, \
                 writer=None, resume=None, output_path=None):

        self.max_step = max_step # The maximum number of steps to run the policy for in each episode.
        self.env_batch = env_batch # The number of environments to run the policy in at the same time.
        self.batch_size = batch_size # The number of training examples to process before updating the model parameters.            

        # Initializing Actor network
        self.actor = ResNet(9, 18, 65) # target, canvas, stepnum, coordconv 3 + 3 + 1 + 2

        # Initializing target network for actor
        self.actor_target = ResNet(9, 18, 65)

        # Initializing critic network
        self.critic = ResNet_wobn(3 + 9, 18, 1) # add the last canvas for better prediction

        # Initializing target network for critic
        self.critic_target = ResNet_wobn(3 + 9, 18, 1) 

        # Optimizer based on stochastic gradient descent
        self.actor_optim  = Adam(self.actor.parameters(), lr=1e-2)
        self.critic_optim  = Adam(self.critic.parameters(), lr=1e-2)

        # updates online actor and critic
        if (resume != None):
            self.load_weights(resume)

        # A hard update involves directly assigning the updated weights and biases of the model to its corresponding parameters. 
        # This means that the model's parameters are abruptly replaced with the new values, discarding any previous information from the old parameters.

        # Hard updating from actor/critic networks to target actor/critic networks
        hard_update(self.actor_target, self.actor)
        hard_update(self.critic_target, self.critic)
        
        # Create replay buffer
        self.memory = rpm(rmsize * max_step)

        # Hyper-parameters
        self.tau = tau
        self.discount = discount

        # Tensorboard
        self.writer = writer
        self.log = 0
        
        self.state = [None] * self.env_batch # Most recent state
        self.action = [None] * self.env_batch # Most recent action
        self.choose_device()        

    # To preict action from either the actor or the target actor network
    def play(self, state, target=False):
        """Predicts action for given current state from either actor or target actor network based on the boolean parameter target"""
        # Normalizing the current state and concatenating it with the coordinates of current state
        state = torch.cat((state[:, :6].float() / 255, state[:, 6:7].float() / self.max_step, coord.expand(state.shape[0], 2, 128, 128)), 1)
        # if target flag is true the action will be predicted from the target actor network
        if target:
            return self.actor_target(state)
        else:
            return self.actor(state)

    # Updates the Generative Adversarial Network (GAN)
    # This network is used to compare real images with fake and calculate losses
    def update_gan(self, state):
        canvas = state[:, :3]
        gt = state[:, 3 : 6]
        fake, real, penal = update(canvas.float() / 255, gt.float() / 255)
        if self.log % 20 == 0:
            self.writer.add_scalar('train/gan_fake', fake, self.log)
            self.writer.add_scalar('train/gan_real', real, self.log)
            self.writer.add_scalar('train/gan_penal', penal, self.log)   
                
    # evaluates the action taken by the agent in a given state and returns the expected reward and the GAN reward 
    def evaluate(self, state, action, target=False):
        """Returns [Q(state, action), gan_reward]
        here Q(state, action) is return from critic_network + gan_reward
        """
        T = state[:, 6 : 7]
        gt = state[:, 3 : 6].float() / 255
        canvas0 = state[:, :3].float() / 255
        canvas1 = decode(action, canvas0)
        # gan reward is the difference b/w real and fake image, gan_reward ~ to similarity b/w fake and real image
        gan_reward = cal_reward(canvas1, gt) - cal_reward(canvas0, gt)
        # L2_reward = ((canvas0 - gt) ** 2).mean(1).mean(1).mean(1) - ((canvas1 - gt) ** 2).mean(1).mean(1).mean(1)        
        coord_ = coord.expand(state.shape[0], 2, 128, 128)
        merged_state = torch.cat([canvas0, canvas1, gt, (T + 1).float() / self.max_step, coord_], 1)
        # canvas0 is not necessarily added
        if target:
            Q = self.critic_target(merged_state)
            return (Q + gan_reward), gan_reward
        else:
            Q = self.critic(merged_state)
            if self.log % 20 == 0:
                self.writer.add_scalar('train/expect_reward', Q.mean(), self.log)
                self.writer.add_scalar('train/gan_reward', gan_reward.mean(), self.log)
            return (Q + gan_reward), gan_reward
    
    def update_policy(self, lr):
        """updates critic and actor network
        critic based on MSE error(value loss) and actor based on mean value error(policy loss)
    soft updates target critic and actor based on critic and actor"""
        self.log += 1
        
        # Dynamic adjustments of learning rates
        for param_group in self.critic_optim.param_groups:
            param_group['lr'] = lr[0]
        for param_group in self.actor_optim.param_groups:
            param_group['lr'] = lr[1]
            
        # Sample batch
        state, action, reward, \
            next_state, terminal = self.memory.sample_batch(self.batch_size, device)

        # updates the GAN network
        self.update_gan(next_state)
        
        # calculating target Q values
        with torch.no_grad():
            # disable gradient computation, useful for pre-trained model
            next_action = self.play(next_state, True)
            target_q, _ = self.evaluate(next_state, next_action, True)
            target_q = self.discount * ((1 - terminal.float()).view(-1, 1)) * target_q # Q = gamma * Q * (0 if terminal else 1)

        cur_q, step_reward = self.evaluate(state, action) # Q(s, a)
        target_q += step_reward.detach() # (target_q ~ gamma * Q(s', a') + r)
        
        # update critic, loss = MSE error
        value_loss = criterion(cur_q, target_q) # MSE loss
        self.critic.zero_grad() # resets gradient
        value_loss.backward(retain_graph=True) # backpropogation, computes gradient of value_loss wrt to critic
        self.critic_optim.step() # updates parameters of critic using critic_optim

        # update actor, loss = mean value
        action = self.play(state)
        pre_q, _ = self.evaluate(state.detach(), action)
        policy_loss = -pre_q.mean()
        self.actor.zero_grad()
        policy_loss.backward(retain_graph=True)
        self.actor_optim.step()
        
        # Target update (soft update)
        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)

        return -policy_loss, value_loss
    
    # storing the values into replay buffer
    def observe(self, reward, state, done, step):
        """storing the values into replay buffer"""
        s0 = torch.tensor(self.state, device='cpu')
        a = to_tensor(self.action, "cpu")
        r = to_tensor(reward, "cpu")
        s1 = torch.tensor(state, device='cpu')
        d = to_tensor(done.astype('float32'), "cpu")
        for i in range(self.env_batch):
            self.memory.append([s0[i], a[i], r[i], s1[i], d[i]])
        self.state = state

    # adding noise for exploration
    def noise_action(self, noise_factor, state, action):
        """adding noise for exploration"""
        noise = np.zeros(action.shape)
        for i in range(self.env_batch):
            action[i] = action[i] + np.random.normal(0, self.noise_level[i], action.shape[1:]).astype('float32')
        return np.clip(action.astype('float32'), 0, 1)
    
    # selecting action from actor network
    def select_action(self, state, return_fix=False, noise_factor=0):
        """selecting action from actor network"""
        self.eval()
        with torch.no_grad():
            action = self.play(state)
            action = to_numpy(action)
        if noise_factor > 0:        
            action = self.noise_action(noise_factor, state, action)
        self.train()
        self.action = action
        if return_fix:
            return action
        return self.action

    # resets agent's state and noise factor
    def reset(self, obs, factor):
        """prepares agent for next episode
        initialises initial state and noise level"""
        self.state = obs
        self.noise_level = np.random.uniform(0, factor, self.env_batch)

    # loads weights from actor.pkl and critic.pkl for actor critic networks
    def load_weights(self, path):
        """loads wgan and online actor-critic parameters"""
        if path is None: return
        self.actor.load_state_dict(torch.load('{}/actor.pkl'.format(path)))
        self.critic.load_state_dict(torch.load('{}/critic.pkl'.format(path)))
        load_gan(path)
        
    def save_model(self, path):
        """saves parameters of online actor, critic and gan in respective pkl files."""
        self.actor.cpu()
        self.critic.cpu()
        torch.save(self.actor.state_dict(),'{}/actor.pkl'.format(path))
        torch.save(self.critic.state_dict(),'{}/critic.pkl'.format(path))
        save_gan(path)
        self.choose_device()

    # This function sets the agent's networks to evaluation mode.
    # This means that the agent will not use dropout or other regularization techniques when making predictions.
    # By disabling dropout and regularization, the agent is able to make predictions without any of the noise or uncertainty that these techniques can introduce.
    def eval(self):
        self.actor.eval()
        self.actor_target.eval()
        self.critic.eval()
        self.critic_target.eval()
    
    # This function sets the agent's networks to training mode.
    # This means that the agent will use dropout or other regularization techniques when making predictions.
    def train(self):
        self.actor.train()
        self.actor_target.train()
        self.critic.train()
        self.critic_target.train()
    # This function moves the agent's networks to the desired device (CPU or GPU).
    def choose_device(self):
        Decoder.to(device)
        self.actor.to(device)
        self.actor_target.to(device)
        self.critic.to(device)
        self.critic_target.to(device)
