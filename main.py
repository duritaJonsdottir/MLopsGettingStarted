import argparse
import sys

import torch

from data import mnist
from model import MyAwesomeModel
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt


class TrainOREvaluate(object):
    """ Helper class that will help launch class methods as commands
        from a single script
    """
    def __init__(self):
        parser = argparse.ArgumentParser(
            description="Script for either training or evaluating",
            usage="python main.py <command>"
        )
        parser.add_argument("command", help="Subcommand to run")
        args = parser.parse_args(sys.argv[1:2])
        if not hasattr(self, args.command):
            print('Unrecognized command')
            
            parser.print_help()
            exit(1)
        # use dispatch pattern to invoke method with same name
        getattr(self, args.command)()
    
    def train(self):
        print("Training day and night")
        parser = argparse.ArgumentParser(description='Training arguments')
        parser.add_argument('--lr', default=0.1)
        # add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])
        print(args)
        
        # TODO: Implement training loop here
        model = MyAwesomeModel() # My model
        criterion = torch.nn.CrossEntropyLoss()  # COst function
        optimizer = optim.SGD(model.parameters(), lr=0.01)  # Optimizer
        
        # Dataset
        train_set, _ = mnist()
        train_dataset = TensorDataset(torch.Tensor(train_set['images']), torch.Tensor(train_set['labels']))
        trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=4,shuffle=True, num_workers=2)

        # Define varibles
        num_epoch = 7
        num_samples = len(train_dataset)
        print(num_samples)

        loss = []
        for epoch in range(num_epoch):  # loop over the dataset multiple times
            for i, data in enumerate(trainloader, 0):
                # get the inputs, batch size of 4
                batch, labels = data

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                batch = batch[:, None, :, :]
                output = model(batch)

                # Recontruct labels
                lab = torch.zeros(4, 10)
                for i in range(0,len(labels)):
                    lab[i,int(labels[i])] = 1
                

                batch_loss = criterion(output[0], lab)
                batch_loss.backward()
                optimizer.step()
                loss.append(batch_loss.detach().numpy())


        plt.plot(loss)
        plt.ylabel('Loss')
        plt.show()
        torch.save(model.state_dict(), 'model2.pt')
        print('Finished Training')
        


    def evaluate(self):
        print("Evaluating until hitting the ceiling")
        parser = argparse.ArgumentParser(description='Training arguments')
        parser.add_argument('load_model_from', default="")
        # add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])
        print(args)
        
    
        model = MyAwesomeModel()
        model.load_state_dict(torch.load(args.load_model_from))
        model.eval()

        # Dataset
        _, test_set = mnist()
        test_dataset = TensorDataset(torch.Tensor(test_set['images']), torch.Tensor(test_set['labels']))
        testloader = torch.utils.data.DataLoader(test_dataset, batch_size=1,shuffle=True, num_workers=2)

        correct = 0
        total = 0

        for data in testloader:
            images, labels = data
            images = images[None, :, :, :]
            outputs = model(images)
            _, predicted = torch.max(outputs[0].data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()

        print('Accuracy of the network on the {} test images: {:4.2f} %'.format(
            test_set['images'].shape[0], 100 * correct.true_divide(total)))

if __name__ == '__main__':
    TrainOREvaluate()
    
    
    
    
    
    
    
    
    