import copy
import torch

from torch import optim, nn
from collections import defaultdict
from torch.utils.data import DataLoader

from utils.utils import HardNegativeMining, MeanReduction, unNormalize
import os
import matplotlib.pyplot as plt
from torch.optim import lr_scheduler
import numpy as np
from matplotlib.patches import Rectangle
import wandb



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


    def run_epoch(self, cur_epoch, n_steps: int) -> None:
        '''
        This method locally trains the model with the dataset of the client. It handles the training at mini-batch level.
            -) cur_epoch: current epoch of training;
            -) optimizer: optimizer used for the local training.
        '''

        print('epoch', cur_epoch + 1)
        for cur_step, (images, labels) in enumerate(self.train_loader):
                
            # Total steps needed to complete an epoch. Computed as:
            # 
            #           self.n_total_steps = floor(len(self.dataset) / self.args.bs).
            # 
            # Example: len(self.dataset) = 600, self.args.bs = 16, self.n_total_steps = 37
            self.n_total_steps = len(self.train_loader)
            
            images = images.to(self.device) 
            labels = labels.to(self.device)
            outputs = self._get_outputs(images)
            
            loss = self.criterion(outputs, labels.long())
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
            # We keep track of the loss. Notice we are storing the loss for
            # each mini-batch.
            wandb.log({"loss": loss.mean()})
            
            # We are considering 10 batch at a time. TO DO: define a way to handle different values.
            # We are considering n_steps batch at a time.
            if (cur_step + 1) % n_steps == 0 or cur_step + 1 == self.n_total_steps:

                # We store the information about the steps using self.count. This is needed if we want to plot the learning curve.
                if (cur_step + 1) % n_steps == 0:
                    self.count += n_steps
                else:
                    self.count += (cur_step + 1) % n_steps  # We need to consider the special case in which we have a number of batches
                                                            # different from the steps we fixed (e.g. each 10 steps, but only 7 left).
                
                # We store the values of the mean loss and of the std each n_steps mini-batches (e.g. mean loss of the 10th mini-batch).
                self.mean_loss.append(loss.mean().cpu().detach().numpy())
                self.mean_std.append(loss.std().cpu().detach().numpy())
                
                # We store the information related to the step in which we computed the loss of the n_steps-th mini-batch. This will
                # be of use when plotting the learning curve.
                self.n_10th_steps.append(self.count)
                print(f'epoch {cur_epoch + 1} / {self.args.num_epochs}, step {cur_step + 1} / {self.n_total_steps}, loss = {loss.mean():.3f} ± {(loss.std() / np.sqrt(self.args.bs)):.3f}')


    def set_opt(self, params: dict) -> None:
        '''
        This helper function supports us when passing the optimization hyperparameters.
          -) the optimization algorithm and its parameters;
          -) the rate decay and its parameters.
        '''
        
        # Get parameters and retrieve the methods desidered by the user.
        self.params = params

        # We extract the names, we'll need them later to extract the methods as well.
        self.opt, self.sch = params['optimizer']['name'], params['scheduler']['name']
        self.opt_method, self.sch_method = getattr(optim, self.opt), getattr(lr_scheduler, self.sch)


    def print_learning(self, step: int, plot_error = False) -> None:
        '''
        Function used to print the learning curves.
        -) step: how many steps we want to wait until plotting the loss;
        -) plot_error: whether we want to plot the std of the loss.
        '''
        
        # We need arrays in order to plot the results.
        self.mean_loss = np.array(self.mean_loss)
        self.mean_std  = np.array(self.mean_std)

        # We plot vertical lines for each epoch. Notice that the epoch may not be a multiple of the argument step, since
        # len(self.dataset) may not be perfectly divisible by self.args.bs (e.g. step = 10, self.n_total_steps = 37).
        lines = [plt.axvline(_x, linewidth = 1, color = 'g', alpha = 0.75,
                            linestyle = '--') for _x in self.n_epoch_steps]
        
        # We plot the loss curve uing plt.errorbar since we may want to include the std as well.
        markers, caps, bars = plt.errorbar(self.n_10th_steps, self.mean_loss, 
                                            yerr = self.mean_std / np.sqrt(self.args.bs),   # We consider the SE of the mean.
                                            ecolor = 'red', elinewidth = 1.5, 
                                            capsize = 1.5, capthick = 1.0, 
                                            color = 'blue',
                                            marker = 'o', fillstyle = 'none')
        plt.title('Loss')

        # This was done only to plot extra text in the legend, namely the optimizer and the rate decay method we chose.
        extra = Rectangle((0, 0), 1, 1, fc = "w", fill = False, 
                            edgecolor = 'none', linewidth = 0)
        text = f'Optimizer: {self.opt}\nRate decay: {self.sch}'
        plt.legend([extra, markers, lines[0]], (text, 'Loss ± SE', 'Epoch'))

        # We plot a line for each step (e.g. one line each n_steps mini-batches).
        xticks = np.arange(step, self.args.num_epochs * np.floor(len(self.dataset) / self.args.bs) , step)
        plt.xticks(xticks)
        plt.grid(axis = 'x', alpha = 0.5)
        
        # We do this in order to consider whether to plot the errors as well.
        [bar.set_alpha(0.3 * plot_error) for bar in bars]
        [cap.set_alpha(0.3 * plot_error) for cap in caps]

        plt.show()


    def train(self, n_steps: int):
        '''
        This method locally trains the model with the dataset of the client. It handles the training at epochs level
        (by calling the run_epoch method for each local epoch of training)
        :return: length of the local dataset, copy of the model parameters
        '''
        
        self.model.train()
        
        # Freeze parameters so we don't backprop through them.
        for param in self.model.backbone.parameters():
            param.requires_grad = False
        print('Params freezed')

        # We build the effective optimizer and scheduler. We need first to create fake dictionaries to pass as argument.
        dummy_dict = {'params': self.model.classifier.parameters()}
        opt_param = self.params['optimizer']['settings']
        dummy_dict.update(opt_param)
        self.optimizer = self.opt_method([dummy_dict])

        dummy_dict = {'optimizer': self.optimizer}
        sch_param = self.params['scheduler']['settings']
        dummy_dict.update(sch_param)
        self.scheduler = self.sch_method(**dummy_dict)


        # Training loop. We initialize some empty lists because we need to store the information about the statistics computed
        # on the mini-batches.
        self.n_total_steps = len(self.train_loader)
        self.mean_loss = []
        self.mean_std  = []
        self.n_10th_steps = []
        self.n_epoch_steps = [self.n_total_steps]
        self.count = 0

        # We initialize a run. We define the name of the project
        # and the configuration, as well as some notes and tags.
        run = wandb.init(     
                                 
          # Set the project where this run will be logged
          project = "testing",                                # We create a project with a given name.
          
          # Track hyperparameters and run metadata
          config = self.params,

          notes = "My first experiment",                      # We can add notes...
          tags = ["baseline", "paper1"]                       # ...and tags as well.
          )

        # We iterate over the epochs.
        for epoch in range(self.args.num_epochs):

            self.run_epoch(epoch, n_steps)
            self.scheduler.step()
                
            # Here we are simply computing how many steps do we need to complete an epoch.
            self.n_epoch_steps.append(self.n_epoch_steps[0] * (epoch + 1))
                
        
        print('Training finished!')
        torch.save(self.model.classifier.state_dict(), 'modelliSalvati/checkpoint.pth')
        print('Model saved!')


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
    
    def checkRndImageAndLabel(self, alpha = 0.4):
        # TODO: abbellire la funzione stampando bordi ed etichette
        self.model.eval()
        with torch.no_grad():
            rnd = torch.randint(low = 0, high = 600, size = (1,)).item()
            image = self.dataset[rnd][0]
            outputLogit = self.model(image.view(1, 3, 512, 928))['out'][0]
            prediction = outputLogit.argmax(0)
            plt.imshow(unNormalize(image[0].cpu()).permute(1,2,0))
            plt.imshow(prediction.cpu().numpy(), alpha = alpha)