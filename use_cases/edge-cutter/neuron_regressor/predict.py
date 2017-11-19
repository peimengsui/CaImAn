import torch
import pickle
import vgg
import numpy as np
from data_loader import *
import pandas as pd
from neuron_dataset import NeuronDataset, TestDataset
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


model_dict = torch.load('save_temp/checkpoint_64.tar')['state_dict']
regressor = vgg.__dict__['vgg11_bn']()
regressor.load_state_dict(model_dict)
regressor = regressor.cpu()

#test_dataset = NeuronDataset(label_file='/mnt/ceph/neuro/edge_cutter/test_images/all_labels_test.pkl',
#                                           image_dir='/mnt/ceph/neuro/edge_cutter/test_images',
#                                           transform=transforms.Compose([
#                                               transforms.ToTensor()
#                                           ]))
#test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

test_dataset = TestDataset(label_file='/mnt/ceph/neuro/edge_cutter/test_crops/valid_label_dic.pkl',
                                                                                   image_dir='/mnt/ceph/neuro/edge_cutter/test_crops/',
                                                                                   transform=transforms.Compose([
                                                                                           transforms.ToTensor()
                                                                                   ]))
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4)
def predict(test_loader, model):
    """
    Run evaluation
    """
    # switch to evaluate mode
    data_list = []
    prediction = []
    ground_truth = []
    model.eval()

    n_val = 0
    #for i, (input, target) in enumerate(val_loader):
    for i, (input, target) in enumerate(test_loader):
        data_list.append(input.numpy().reshape((128, 64, 64)))
        ground_truth += [int(x) for x in target.numpy().flatten()]
        input_var = torch.autograd.Variable(input, volatile=True)
        #if torch.cuda.is_available():
        #    input_var = input_var.cuda()

        # compute output
        prediction += [float(x) for x in model(input_var).cpu().data.numpy().flatten()]
        
        n_val += 1
        if n_val > 60:
            break


    return data_list, prediction, ground_truth

data_list, prediction, ground_truth = predict(test_loader, regressor)
pickle.dump(data_list, open('save_temp/prediction_images.pkl',"wb"))
df = pd.DataFrame({'prediction':prediction,'ground_truth':ground_truth})
df.to_csv('save_temp/prediction.csv')

