import torch
import pickle

regressor = torch.load('save_temp/checkpoint_63.tar')
import pdb
pdb.set_trace()

no_neuron_image = np.load('/mnt/ceph/neuro/edge_cutter/25_zero_input_data/Yr_d1_512_d2_512_d3_1_order_C_frames_8000_.test.npz')
no_neuron_image = no_neuron_image['arr_0']

no_neuron_locations = pickle.load(open('/mnt/ceph/neuro/edge_cutter/25_zero_input_data/Yr_d1_512_d2_512_d3_1_order_C_frames_8000_.test.pkl', 'rb'))

test_image  = np.load('/mnt/ceph/neuro/edge_cutter/25_input_data/Yr_d1_512_d2_512_d3_1_order_C_frames_8000_.test.npz')
test_image = test_image['arr_0']
test_locations = pickle.load(open('/mnt/ceph/neuro/edge_cutter/25_input_data/Yr_d1_512_d2_512_d3_1_order_C_frames_8000_.test.pkl', 'rb')) 

test_loader = data_generator(test_image, test_locations, no_neuron_image, no_neuron_locations, batch_size=1, train=False)

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
    for i, (input, target) in val_loader:
        data_list.append(input.numpy())
        ground_truth.append(int(target.numpy()))
        input_var = torch.autograd.Variable(input, volatile=True)
        if torch.cuda.is_available():
            input_var = input_var.cuda()

        # compute output
        prediction.append(float(model(input_var).cpu().data.numpy()))
        
        n_val += 1
        if n_val > 1024:
            break


    return data_list, prediction, ground_truth

data_list, prediction, ground_truth = predict(test_loader, regressor)
pickle.dump(data_list, open('save_temp/prediction_images.pkl'))


