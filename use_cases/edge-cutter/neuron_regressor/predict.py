import torch
import pickle
import vgg
import numpy as np
from data_loader import *
import pandas as pd

model_dict = torch.load('save_temp/checkpoint_63.tar')['state_dict']
regressor = vgg.__dict__['vgg11_bn']()
regressor.load_state_dict(model_dict)
regressor = regressor.cpu()
no_neuron_image = np.load('/mnt/ceph/neuro/edge_cutter/25_zero_input_data/Yr_d1_512_d2_512_d3_1_order_C_frames_8000_.test.npz')
no_neuron_image = no_neuron_image['arr_0']

no_neuron_locations = pickle.load(open('/mnt/ceph/neuro/edge_cutter/25_zero_input_data/Yr_d1_512_d2_512_d3_1_order_C_frames_8000_.test.pkl', 'rb'))

test_image  = np.load('/mnt/ceph/neuro/edge_cutter/25_input_data/Yr_d1_512_d2_512_d3_1_order_C_frames_8000_.test.npz')
test_image = test_image['arr_0']
test_locations = pickle.load(open('/mnt/ceph/neuro/edge_cutter/25_input_data/Yr_d1_512_d2_512_d3_1_order_C_frames_8000_.test.pkl', 'rb')) 

test_loader = data_generator(test_image, test_locations, no_neuron_image, no_neuron_locations, batch_size=128, train=False)

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
    for i, (input, target) in test_loader:
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

