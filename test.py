import torch
from configuration import conf
from utils.data_loader import load_mnist
from models.base import Linear_base_model, Convolutional_base_model
from utils.visualise import *


#model.train(trainloader)

# Load the model e.g. model.load_state_dict(torch.load('./ckp/mnist_PNN_96.92.pt'))

# Load the DE model

conf.num_distr = '2'
conf.layer_type = 'DE'

trainloader, testloader = load_mnist()

if conf.model_type == 'CNN':
    model = Convolutional_base_model()
elif conf.model_type == 'NN':
    model = Linear_base_model()

print(model)

model.load_state_dict(torch.load('./ckp/NN/mnist_DE_96.03.pt'))
model.test(testloader)
