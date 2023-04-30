import copy
import torch

from torch import optim, nn
from collections import defaultdict
from torch.utils.data import DataLoader

from utils.utils import HardNegativeMining, MeanReduction, unNormalize
import os
import matplotlib.pyplot as plt





class Centralized:

    def __init__(self, args, dataset, model, test_client=False):
        self.args = args
        self.dataset = dataset
        self.name = self.dataset.client_name
        self.model = model
        #! da rimuovere se si passa dal main 
        self.model.cuda()
        self.train_loader = DataLoader(self.dataset, batch_size=self.args.bs, shuffle=True, drop_last=True) \
            if not test_client else None
        self.test_loader = DataLoader(self.dataset, batch_size=1, shuffle=False)
        self.criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='none')
        self.reduction = HardNegativeMining() if self.args.hnm else MeanReduction()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    @staticmethod
    def updatemetric(metric, outputs, labels):
        _ , prediction = outputs.max(dim=1)
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

    #TODO hyperparameter tuning
    """
    Perprocess:
        Transformations
            RandomHorizontalFlip
            ColorJitter
            RandomScaleRandomCrop
            RandomResizedCrop
    
    training:
        lr = [1e-4, 1e-1]
        optimizer = [adam, sgd, adadelta]
        bs = [16, 32, 600]
        dropout rate = [0.5, 0.8]
    """

    def train(self):
        """
        This method locally trains the model with the dataset of the client. It handles the training at epochs level
        (by calling the run_epoch method for each local epoch of training)
        :return: length of the local dataset, copy of the model parameters
        """
        # define loss and optimizer
        self.model.train()
        # Freeze parameters so we don't backprop through them
        for param in self.model.backbone.parameters():
            param.requires_grad = False
        print('params freezed')

        optimizer = optim.SGD(self.model.classifier.parameters(), lr=0.0001, momentum=0.9)
        # Training loop
        n_total_steps = len(self.train_loader)
        for epoch in range(self.args.num_epochs):
            print("epoca", epoch)
            for i, (images,labels) in enumerate(self.train_loader):
                images = images.to(self.device, dtype = torch.float32) 
                labels = labels.to(self.device, dtype = torch.long)
                outputs = self._get_outputs(images)
                loss = self.reduction(self.criterion(outputs,labels), labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (i+1) % 10 == 0 or i+1 == n_total_steps:
                    print(f'epoch {epoch+1} / {self.args.num_epochs}, step {i+1} / {n_total_steps}, loss = {loss.mean():.3f}')

        print("Finish training")
        #torch.save(self.model.classifier.state_dict(), 'modelliSalvati/checkpoint.pth')
        #print("Model saved")


    #i dati vengono testati sugli stessi dati di training
    def test(self, metric):
        """
        This method tests the model on the local dataset of the client.
        :param metric: StreamMetric object
        """
        self.model.eval()
        with torch.no_grad():
            for i, (images, labels) in enumerate(self.test_loader):
                images = images.to(self.device) 
                labels = labels.to(self.device)
                outputs = self._get_outputs(images)
                self.updatemetric(metric, outputs, labels)
    
    #TODO: da far funzionare
    def checkRndImageAndLabel(self, alpha = 0.4):
        # TODO: abbellire la funzione stampando bordi ed etichette
        self.model.eval()
        with torch.no_grad():
            rnd = torch.randint(low = 0, high = 600, size = (1,)).item()
            image = self.dataset[rnd][0].cuda()
            outputLogit = self.model(image.view(1, 3, 512, 928))['out'][0]
            prediction = outputLogit.argmax(0)
            plt.imshow(unNormalize(image[0].cpu()).permute(1,2,0))
            plt.imshow(prediction.cpu().numpy(), alpha = alpha)