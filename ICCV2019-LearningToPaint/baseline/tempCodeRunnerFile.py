# exp = os.path.abspath('.').split('/')[-1]

# # Tensorboard writer to log the training process
# writer = tensorboardX.SummaryWriter(os.path.join('train_log', exp))
# # symbolic link to the training log directory
# os.system('ln -sf ../train_log/{} ./log'.format(exp))
# # creating directory to store trained model
# os.system('mkdir ./model')

# to get the current experience name
exp = os.path.basename(os.path.abspath('.'))
# Tensorboard writer to log the training process
writer = tensorboardX.SummaryWriter(os.path.join('train_log', exp))
# symbolic link to the training log directory
os.system('mklink /D .\\log ..\\train_log\\{}'.format(exp))
# creating directory to store trained model
os.system('mkdir .\\model',exist_ok=True)