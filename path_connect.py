import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import os
import logging
import numpy as np
from scipy.special import softmax

from models.base import Linear_base_model, Convolutional_base_model
from configuration import conf

from art.attacks import FastGradientMethod, BasicIterativeMethod, ProjectedGradientDescent
from art.classifiers import PyTorchClassifier
from art.utils import load_mnist as art_load_mnist
from art.utils import compute_success, get_labels_np_array, check_and_transform_label_format
from art.config import ART_NUMPY_DTYPE

np.random.seed(2020)
logger = logging.getLogger(__name__)


class MinimalPerturbationIterativeMethod(BasicIterativeMethod):

    def generate(self, x, y=None, **kwargs):
        """
        Generate adversarial samples and return them in an array.

        :param x: An array with the original inputs.
        :type x: `np.ndarray`
        :param y: Target values (class labels) one-hot-encoded of shape `(nb_samples, nb_classes)` or indices of shape
                  (nb_samples,). Only provide this parameter if you'd like to use true labels when crafting adversarial
                  samples. Otherwise, model predictions are used as labels to avoid the "label leaking" effect
                  (explained in this paper: https://arxiv.org/abs/1611.01236). Default is `None`.
        :type y: `np.ndarray`
        :return: An array holding the adversarial examples.
        :rtype: `np.ndarray`
        """
        assert x.shape[0] == 1, "x.shape[0] != 1: Minimal perturbation calculation only works for individual sample. "
        y = check_and_transform_label_format(y, self.classifier.nb_classes())

        if y is None:
            # Throw error if attack is targeted, but no targets are provided
            if self.targeted:
                raise ValueError("Target labels `y` need to be provided for a targeted attack.")

            # Use model predictions as correct outputs
            targets = get_labels_np_array(self.classifier.predict(x, batch_size=self.batch_size))
        else:
            targets = y

        adv_x_best = None
        rate_best = None

        if self.random_eps:
            ratio = self.eps_step / self.eps
            self.eps = np.round(self.norm_dist.rvs(1)[0], 10)
            self.eps_step = ratio * self.eps

        for _ in range(max(1, self.num_random_init)):
            adv_x = x.astype(ART_NUMPY_DTYPE)

            for i_max_iter in range(self.max_iter):
                adv_x = self._compute(
                    adv_x,
                    x,
                    targets,
                    self.eps,
                    self.eps_step,
                    self._project,
                    self.num_random_init > 0 and i_max_iter == 0,
                )

                pred = np.argmax(self.classifier.predict(adv_x), axis=1)[0]
                if pred == np.argmax(targets[0]):
                    print('Target achieved at iteration %d' % i_max_iter)
                    break

            if self.num_random_init > 1:
                rate = 100 * compute_success(
                    self.classifier, x, targets, adv_x, self.targeted, batch_size=self.batch_size
                )
                if rate_best is None or rate > rate_best or adv_x_best is None:
                    rate_best = rate
                    adv_x_best = adv_x
            else:
                adv_x_best = adv_x

        return adv_x_best


try:
    import cPickle as pickle
except ImportError:  # python 3.x
    import pickle


class BCEOneHotLoss(nn.BCELoss):
    def forward(self, input, target):
        return super(BCEOneHotLoss, self).forward(input, target.float())


def compute_confidence(preds, head='FC'):
    return np.max(softmax(preds, axis=1), axis=1) if head == 'FC' else np.max(preds, axis=1)


# Load pretrained model
HEAD = 'FC'
FEATURE = 'NN'
NUM_DISTR = 'num_distr=1'


model_directory = os.path.join('ckp', NUM_DISTR, FEATURE)
conf.num_distr = NUM_DISTR[-1]

if FEATURE == 'CNN':
    conf.model_type, conf.hidden_units = 'CNN', '100'
else:
    conf.model_type, conf.hidden_units = 'NN', '784,200,200'

if HEAD == 'DE':
    CHECKPOINT = os.path.join(model_directory, 'mnist_DE.pt')
    conf.layer_type = 'DE'
    criterion = BCEOneHotLoss()
elif HEAD == 'PNN':
    CHECKPOINT = os.path.join(model_directory, 'mnist_PNN.pt')
    conf.layer_type = 'PNN'
    criterion = BCEOneHotLoss()
else:
    conf.layer_type = 'FC'
    CHECKPOINT = os.path.join(model_directory, 'mnist_FC.pt')
    criterion = nn.CrossEntropyLoss()


model = Convolutional_base_model() if FEATURE == 'CNN' else Linear_base_model()
checkpoint = torch.load(CHECKPOINT)
model.load_state_dict(checkpoint)

# Step 1: Load the MNIST dataset
(x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = art_load_mnist()
# Step 1a: Transpose to N x D format
if type(model) == Linear_base_model:
    x_train = x_train.reshape((x_train.shape[0], -1)).astype(np.float32)
    x_test = x_test.reshape((x_test.shape[0], -1)).astype(np.float32)
else:
    x_train = x_train.transpose((0, 3, 1, 2)).astype(np.float32)
    x_test = x_test.transpose((0, 3, 1, 2)).astype(np.float32)

# Step 2a: Define the optimizer
optimizer = optim.Adam(model.parameters())

# Step 3: Create the ART classifier
classifier = PyTorchClassifier(
    model=model,
    clip_values=(min_pixel_value, max_pixel_value),
    loss=criterion,
    optimizer=optimizer,
    input_shape=(784, ) if type(model) == Linear_base_model else (1, 28, 28),
    nb_classes=10,
    preprocessing=(0.1307, 0.3081)
)

# Define attack
eps = 0.01
attack = MinimalPerturbationIterativeMethod(classifier=classifier, eps=eps, eps_step=eps/3, targeted=True, max_iter=999)


def get_data_pairs(xs, num=1):
    data_pairs = []
    predictions = np.argmax(classifier.predict(xs), axis=1)
    labels = np.unique(predictions)
    for label in labels:
        indices = np.argwhere(predictions == label).squeeze()
        for i in range(num):
            indices = np.random.permutation(indices)
            data_pairs.append((xs[indices[0], ...], xs[indices[-1], ...]))
    np.random.shuffle(data_pairs)
    return data_pairs


def check_path_validity(xstart, xend, label, convex_comb=4):
    step_vec = (xend - xstart) / (convex_comb + 1)
    anchors = []
    for i in range(1, convex_comb + 1):
        anchors.append(xstart + step_vec * i)
    anchors = np.stack(anchors)
    preds = np.argmax(classifier.predict(anchors), axis=1)
    unique_preds = np.unique(preds)
    return False if len(unique_preds) != 1 or unique_preds[0] != label else True


def find_path(x1, x2, degree=0):
    if degree > 10:
        print("No connected path found within 10 recursion")
        return [x1, x2]

    x = np.stack((x1, x2))
    preds = np.argmax(classifier.predict(x), axis=1)
    if len(np.unique(preds)) > 1:
        print("Input samples belong to different categories according to the classifier")
        return []
    else:
        label = np.unique(preds)[0]

    xm = (x1 + x2) / 2
    x_ = np.expand_dims(xm, axis=0)
    pred_m = np.argmax(classifier.predict(x_), axis=1)[0]
    if pred_m != label:
        params = {'y': torch.eye(classifier.nb_classes())[[label]].numpy()}
        x_adv = attack.generate(x_, **params)
        xm = x_adv.squeeze()

        if np.argmax(classifier.predict(x_adv), axis=1)[0] != label:
            print("No connected path found because the mid-point cannot be perturbed to the target category")
            return []

    # Check the validity of the path and keep finding recursively when it's invalid
    path1 = [x1, xm] if check_path_validity(x1, xm, label) else find_path(x1, xm, degree + 1)
    path2 = [xm, x2] if check_path_validity(xm, x2, label) else find_path(xm, x2, degree + 1)
    return path1 + path2[1:]


# Step 4: Select data pairs from the same category
data_pairs = get_data_pairs(x_test, num=1000)

# Step 5: Find a path that connects the two inputs with same predictions
for idx, data_pair in enumerate(data_pairs):
    print("Index: %d" % idx)
    path = find_path(data_pair[0], data_pair[-1])
    print(len(path))
pass
