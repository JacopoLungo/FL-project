import copy
from collections import OrderedDict

import numpy as np
import torch


class Server:

    def __init__(self, args, train_clients, test_clients, model, metrics):
        self.args = args
        self.train_clients = train_clients #lista con un solo elemento (istanza della classe centralized (test_client = False))
        self.test_clients = test_clients #lista con due elementi (istanza della classe centralized (test_client = True))
        self.model = model #da passare poi al client
        self.metrics = metrics
        self.model_params_dict = copy.deepcopy(self.model.state_dict())
        #per setting centralized
        self.params = None

    def select_clients(self):
        num_clients = min(self.args.clients_per_round, len(self.train_clients))
        return np.random.choice(self.train_clients, num_clients, replace=False)

    def train_round(self, clients):
        """
            This method trains the model with the dataset of the clients. It handles the training at single round level
            :param clients: list of all the clients to train
            :return: model updates gathered from the clients, to be aggregated
        """
        """
        Fa il train sul trainClient (train.txt)
        """
        updates = []
        for i, c in enumerate(clients):
            # TODO: missing code here!
            raise NotImplementedError
        return updates #nel setting centralized restituisce direttamente i pesi

    def aggregate(self, updates):
        """
        This method handles the FedAvg aggregation
        :param updates: updates received from the clients
        :return: aggregated parameters
        """
        # TODO: missing code here!
        raise NotImplementedError

    def train(self):
        """
        This method orchestrates the training the evals and tests at rounds level
        """
        """
        Train chiama train_round(), salva il modello restituito da train_round(),
        chiama eval_train (train.txt) che fa l'evaluation sul train client e test()
        che fa l'evaluation su entrambi i test clients (test_same_dom e test_diff_dom)
        """
        for r in range(self.args.num_rounds):
            #
            self.parmas = self.train_round(self.train_clients[0])
            state_dict  = torch.load('modelliSalvati/checkpoint.pth')
            self.model.classifier.load_state_dict(state_dict)

            self.eval_train()
            self.test()



            # TODO: missing code here!
            raise NotImplementedError

    def eval_train(self):
        """
        This method handles the evaluation on the train clients
        """
        # TODO: missing code here!
        client = self.train_clients[0]
        print(f"Testing client {client.name}...")
        
        raise NotImplementedError

    def test(self):
        """
            This method handles the test on the test clients
        """
        # TODO: missing code here!

        for client in self.test_clients:
            print(f"Testing client {client.name}...")
            self.load_server_model_on_client(client)


        raise NotImplementedError
    
    def load_server_model_on_client(self, client):
        client.model.load_state_dict(self.model_params_dict)
