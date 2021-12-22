#!/local/data/public/2021/jgc47/.dl/bin/python3
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import argparse
import importlib

def train(epoch):
    global train_losses,train_counter,network,args
    network.train()
    for batch_idx, (data,target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = network(data)
        loss = F.nll_loss(output,target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval==0:
            print(f"Train epoch: {epoch} [{batch_idx*len(data)}/{len(train_loader.dataset)} ({batch_idx*len(data)/len(train_loader.dataset)*100:.2f}%)]\tLoss: {loss.item():.2f}")
            train_losses.append(loss.item())
            train_counter.append(batch_idx*64 + (epoch-1)*len(train_loader.dataset))

    if args.save_results:
        if not os.path.exists(f"./results/{args.exp_name}".strip()):
            os.mkdir(f"./results/{args.exp_name}".strip())

        torch.save(network.state_dict(), f"./results/{args.exp_name}/model.pth")
        torch.save(optimizer.state_dict(), f"./results/{args.exp_name}/optimizer.pth")


def test():
    global test_losses,network,args
    network.eval()
    test_loss =0 
    correct=0
    with torch.no_grad():
        for data,target in test_loader:
            output = network(data)
            test_loss += F.nll_loss(output, target, reduction='mean').item()
            pred = output.data.max(1,keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
        test_loss /= len(test_loader.dataset)
        print(f"\nTest set: Avg. loss: {test_loss:.8f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100*correct / len(test_loader.dataset):.1f}%)\n")
    test_losses.append(test_loss)

def print_params():
    global args
    print(f"# Experiment name: {args.exp_name}")
    print(f"# number of epochs: {args.n_epochs}")
    print(f"# training batch size: {args.batch_size_train}")
    print(f"# testing batch size: {args.batch_size_test}")
    print(f"# optimizer: {args.optimizer_name}")
    print(f"# learning rate: {args.learning_rate}")
    print(f"# nn module: {args.nn_module_name}")
    print(f"# save results: {args.save_results}")

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser("python3 mnist.py")
    parser.add_argument('-n','--n_epochs', type=int,help='number of epochs to train. Default 6.',const=6,nargs='?',default=6)
    parser.add_argument('-t','--batch_size_train',type=int,help='Default 16.',const=16,nargs='?',default=16)
    parser.add_argument('-v','--batch_size_test',type=int,help='Default 1024.',const=1024,nargs='?',default=1024)
    parser.add_argument('-e','--learning_rate',type=float,help='Default 0.01.',const=0.01,nargs='?',default=0.01)
    parser.add_argument('-m','--momentum',type=float,help='Only for SGD and RMSprop. Default 0.',const=0,nargs='?',default=0)
    parser.add_argument('-o','--optimizer_name',type=str,help='Name of optimizer. Defaults to SGD. Supports SGD, Adam, and RMSprop.',const="SGD",nargs='?',default='SGD')
    parser.add_argument('-a','--nn_module_name',type=str,help='path to pytorch nn.Module definition file, in the [pkg].[mod] format required by importlib.import_module(). Defaults to "basenet", which will load the "./basenet.py" file.',const="..basenet",nargs='?',default="..basenet")
    parser.add_argument('-i','--log_interval',type=int,help='print training results every LOG_INTERVAL mini-batches. Defaults to 50. Set to large number to log only once per epoch.',const=50,nargs='?',default=50)
    parser.add_argument('-s','--save_results',action='store_true',help='whether to save state dictionary of network and optimizer')
    parser.add_argument('-x','--exp_name',type=str,help='Optional. Name of experiment. Results will be saved to ./results/{EXP_NAME}. No default.',nargs='?')

    args = parser.parse_args()
    
    nn_module = importlib.import_module(args.nn_module_name,package='.')
    Net = getattr(nn_module, 'Net')

    random_seed=42

    torch.backends.cudnn.enabled=False
    torch.manual_seed(random_seed)

    network = Net()
    if args.optimizer_name=='SGD':
        optimizer = optim.SGD(network.parameters(), lr=args.learning_rate, momentum=args.momentum)
    elif args.optimizer_name=='RMSprop':
        optimizer = optim.RMSprop(network.parameters(), lr=args.learning_rate,momentum=args.momentum)
    elif args.optimizer_name=='Adam':
        optimizer = optim.Adam(network.parameters(), lr=args.learning_rate)
    else:
        print("Warning: unknown optimizer. Defaulting to SGD.\n")
        optimizer = optim.SGD(network.parameters(), lr=args.learning_rate, momentum=args.momentum)

    train_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.1307,),(0.3081,))])), batch_size=args.batch_size_train,shuffle=True)

    test_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.1307,),(0.3081,))])), batch_size=args.batch_size_test,shuffle=True)


    train_losses = []
    train_counter = []
    test_losses = []
    test_counter = [i*len(train_loader.dataset) for i in range(args.n_epochs+1)]

    print_params()

    # training
    test()
    for epoch in range(1, args.n_epochs + 1):
        train(epoch)
        test()
