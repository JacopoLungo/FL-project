import copy
import torch

from torch import optim, nn
from collections import defaultdict
from torch.utils.data import DataLoader

from utils.utils import HardNegativeMining, MeanReduction

class centralized:

    def __init__(self, args, dataset, model):
        self.args = args
        self.dataset = dataset
        self.model = model
        self.train_loader = DataLoader(self.dataset, batch_size=self.args.bs, shuffle=True, drop_last=True)
        self.test_loader = DataLoader(self.dataset, batch_size=1, shuffle=False)
        self.criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='none')
        self.reduction = HardNegativeMining() if self.args.hnm else MeanReduction()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    @staticmethod
    def update_metric(metric, outputs, labels):
        _, prediction = outputs.max(dim=1)
        labels = labels.cpu().numpy()
        prediction = prediction.cpu().numpy()
        metric.update(labels, prediction)

    def _get_outputs(self, images):
        if self.args.model == 'deeplabv3_mobilenetv2':
            return self.model(images)['out']
        if self.args.model == 'resnet18':
            return self.model(images)
        raise NotImplementedError

    def run_epoch(self, cur_epoch, optimizer):
        """
        This method locally trains the model with the dataset of the client. It handles the training at mini-batch level
        :param cur_epoch: current epoch of training
        :param optimizer: optimizer used for the local training
        """
        for cur_step, (images, labels) in enumerate(self.train_loader):
            # TODO: missing code here!
            raise NotImplementedError

    def train(self):
        """
        This method locally trains the model with the dataset of the client. It handles the training at epochs level
        (by calling the run_epoch method for each local epoch of training)
        :return: length of the local dataset, copy of the model parameters
        """
        # TODO: missing code here!
        # define loss and optimizer
        optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        # Training loop
        n_total_steps = len(self.train_loader)
        for epoch in range(self.args.num_epochs):
            for i, (images,labels) in enumerate(self.train_loader):
                images = images.to(self.device) 
                labels = labels.to(self.device)

                #forward pass
                outputs = self._get_outputs(images)
                loss = self.criterion(outputs,labels)

                #backwards
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (i+1) % 200 == 0:
                    print(f'epoch {epoch+1} / {self.args.num_epochs}, step {i+1} / {n_total_steps}, loss = {loss.item():.3f}')

        print("Finish training")


    def test(self, metric):
        """
        This method tests the model on the local dataset of the client.
        :param metric: StreamMetric object
        """
        # TODO: missing code here!
        with torch.no_grad():
            for i, (images, labels) in enumerate(self.test_loader):
                images = images.to(self.device) 
                labels = labels.to(self.device)
                outputs = self._get_outputs(images)
                self.update_metric(metric, outputs, labels)