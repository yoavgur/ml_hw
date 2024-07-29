import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from scipy.stats import norm
from statsmodels.stats.proportion import proportion_confint

def free_adv_train(model, data_tr, criterion, optimizer, lr_scheduler, \
                   eps, device, m=4, epochs=100, batch_size=128, dl_nw=10):
    """
    Free adversarial training, per Shafahi et al.'s work.
    Arguments:
    - model: randomly initialized model
    - data_tr: training dataset
    - criterion: loss function (e.g., nn.CrossEntropyLoss())
    - optimizer: optimizer to be used (e.g., SGD)
    - lr_scheduer: scheduler for updating the learning rate
    - eps: l_inf epsilon to defend against
    - device: device used for training
    - m: # times a batch is repeated
    - epochs: "virtual" number of epochs (equivalent to the number of 
        epochs of standard training)
    - batch_size: training batch_size
    - dl_nw: number of workers in data loader
    Returns:
    - trained model
    """
    # init data loader
    loader_tr = DataLoader(data_tr,
                           batch_size=batch_size,
                           shuffle=True,
                           pin_memory=True,
                           num_workers=dl_nw)
                           

    # init delta (adv. perturbation) - FILL ME
    delta = torch.zeros([batch_size, *data_tr[0][0].shape]).to(device)
    

    # total number of updates - FILL ME
    total_updates = epochs * len(loader_tr)

    # when to update lr
    scheduler_step_iters = int(np.ceil(len(data_tr)/batch_size))

    # train - FILLE ME
    for epoch in range(epochs):
        for i, (inputs, targets) in enumerate(loader_tr, 0):
            # get inputs and labels
            inputs, labels = inputs.to(device), targets.to(device)
            inputs.requires_grad = True

            for _ in range(m):
                # zero the parameter gradients
                optimizer.zero_grad()

                # If batch size is smaller (last batch), adjust delta
                _delta = delta[:inputs.size(0)]

                # forward + backward + optimize
                outputs = model(inputs + _delta)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # update delta
                with torch.no_grad():
                    _delta += eps*torch.sign(inputs.grad)
                    _delta = torch.clamp(_delta, -eps, eps)

                # update original delta
                delta[:inputs.size(0)] = _delta


            # update learning rate
            if (i+1) % scheduler_step_iters == 0:
                lr_scheduler.step()
    
    # done
    return model


class SmoothedModel():
    """
    Use randomized smoothing to find L2 radius around sample x,
    s.t. the classification of x doesn't change within the L2 ball
    around x with probability >= 1-alpha.
    """

    ABSTAIN = -1

    def __init__(self, model, sigma):
        self.model = model
        self.sigma = sigma

    def _sample_under_noise(self, x, n, batch_size):
        """
        Classify input x under noise n times (with batch size 
        equal to batch_size) and return class counts (i.e., an
        array counting how many times each class was assigned the
        max confidence).
        """
        nb_classes = 4
        counts = np.zeros(nb_classes)
        with torch.no_grad():
            for _ in range(0, n, batch_size):
                batch_x = x.repeat(batch_size, 1, 1, 1)
                batch_x += torch.randn_like(batch_x) * self.sigma

                outputs = self.model(batch_x)
                _, preds = torch.max(outputs, 1)
                counts += np.bincount(preds.cpu().numpy(), minlength=nb_classes)

        return counts

    
    def __sample_under_noise(self, x, n, batch_size):
        """
        Classify input x under noise n times (with batch size 
        equal to batch_size) and return class counts (i.e., an
        array counting how many times each class was assigned the
        max confidence).
        """
        nb_classes = 4
        counts = np.zeros(nb_classes)
        with torch.no_grad():
            x_batch = x.repeat(n, 1, 1, 1)
            x_batch += torch.randn_like(x_batch) * self.sigma

            outputs = self.model(x_batch)
            _, preds = torch.max(outputs, 1)
            counts = np.bincount(preds.cpu().numpy(), minlength=nb_classes)

        return counts
        
    def certify(self, x, n0, n, alpha, batch_size):
        """
        Arguments:
        - model: pytorch classification model (preferably, trained with
            Gaussian noise)
        - sigma: Gaussian noise's sigma, for randomized smoothing
        - x: (single) input sample to certify
        - n0: number of samples to find prediction
        - n: number of samples for radius certification
        - alpha: confidence level
        - batch_size: batch size to use for inference
        Outputs:
        - prediction / top class (ABSTAIN in case of abstaining)
        - certified radius (0. in case of abstaining)
        """
        
        # find prediction (top class c) - FILL ME
        counts0 = self._sample_under_noise(x, n0, batch_size)
        c = counts0.argmax()
        
        # compute lower bound on p_c - FILL ME
        counts = self._sample_under_noise(x, n, batch_size)
        p, _ = proportion_confint(counts[c].item(), ((n // batch_size) + 1)*batch_size, alpha=2*alpha, method="beta")
        
        if p > 0.5:
            radius = self.sigma * norm.ppf(p)
        else:
            c = self.ABSTAIN
            radius = 0.0

        # done
        return c, radius
        

class NeuralCleanse:
    """
    A method for detecting and reverse-engineering backdoors.
    """

    def __init__(self, model, dim=(1, 3, 32, 32), lambda_c=0.0005,
                 step_size=0.005, niters=2000):
        """
        Arguments:
        - model: model to test
        - dim: dimensionality of inputs, masks, and triggers
        - lambda_c: constant for balancing Neural Cleanse's objectives
            (l_class + lambda_c*mask_norm)
        - step_size: step size for SGD
        - niters: number of SGD iterations to find the mask and trigger
        """
        self.model = model
        self.dim = dim
        self.lambda_c = lambda_c
        self.niters = niters
        self.step_size = step_size
        self.loss_func = nn.CrossEntropyLoss()

    def find_candidate_backdoor(self, c_t, data_loader, device):
        """
        A method for finding a (potential) backdoor targeting class c_t.
        Arguments:
        - c_t: target class
        - data_loader: DataLoader for test data
        - device: device to run computation
        Outputs:
        - mask: 
        - trigger: 
        """
        # randomly initialize mask and trigger in [0,1] - FILL ME
        mask = torch.rand((self.dim[2], self.dim[3]), device=device, requires_grad=True)
        trigger = torch.rand(self.dim, device=device, requires_grad=True)

        # run self.niters of SGD to find (potential) trigger and mask - FILL ME
        optimizer = torch.optim.Adam([mask, trigger], lr=self.step_size)

        for _ in range(self.niters):
            for inputs, _ in data_loader:
                inputs = inputs.to(device)
                expanded_mask = mask.repeat(inputs.size(1), 1, 1).unsqueeze(0).expand_as(inputs)
                # Apply the trigger and mask to the inputs
                perturbed_inputs = inputs * (1 - expanded_mask) + trigger * expanded_mask

                # Forward pass
                outputs = self.model(perturbed_inputs)

                # Compute the classification loss
                target_labels = torch.full((inputs.size(0),), c_t, device=device, dtype=torch.long)
                loss_class = self.loss_func(outputs, target_labels)

                # Compute the mask norm loss
                mask_norm = mask.abs().sum()
                loss = loss_class + self.lambda_c * mask_norm

                # Backward pass and optimization step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Clamp mask and trigger to [0, 1]
                mask.data.clamp_(0, 1)
                trigger.data.clamp_(0, 1)

        # done
        return mask.repeat(3, 1, 1), trigger
