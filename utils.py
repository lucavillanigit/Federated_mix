#Useful libraries
import copy
import torch
from torchvision import datasets,transforms
from sampling import cifar_iid, cifar_noniid

def exp_details(args):
    print('\nExperimental details:')
    print(f'    Model     : {args.model}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning  : {args.lr}')
    print(f'    Global Rounds   : {args.epochs}\n')

    print('    Federated parameters:')
    if args.iid:
        print('    IID')
    else:
        print('    Non-IID')
    print(f'    Fraction of users  : {args.frac}')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.local_ep}')
    print(f'    Alpha Batch Norm   : {args.alpha_b}')
    print(f'    Alpha Group Norm   : {args.alpha_g}')
    return

def get_dataset(args):
    #[TODO] Add wrapper for multiple datasets
    data_dir = '../data/cifar/'
    
    #Normalize used with mean and stds of Cifar10
    apply_transform = transforms.Compose(
        [transforms.RandomResizedCrop(32),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize([0.49139968, 0.48215841, 0.44653091], [0.24703223, 0.24348513, 0.26158784])])

    
    train_dataset = datasets.CIFAR10(data_dir, train= True,download=True, 
                                   transform=apply_transform)

    test_dataset = datasets.CIFAR10(data_dir,train=False,download=True,
                                  transform=apply_transform)
    
    #sample training
    if args.iid:
        #sample IID user
        user_group = cifar_iid(train_dataset,args.num_users)
    else:
        #sample Non-IID user
        user_group, _ = cifar_noniid(train_dataset,args.num_users)
   
    return train_dataset, test_dataset, user_group 


def average_weights(w, counts):
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        w_avg[key] = 0
        for i in range(len(w)):
            w_avg[key] += torch.mul(w[i][key], counts[i]/sum(counts))
        
    return w_avg