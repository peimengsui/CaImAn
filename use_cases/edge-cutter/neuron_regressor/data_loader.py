import random
import numpy as np
import pickle
import torch

def data_generator(image_arr, locations, no_neuron_image, no_locations, batch_size=32, \
                   b_box_size=5, crop_size=64, train=True):
    '''
    image_arr: (7990, 512, 512)
    locations: [[array([x,y]), array([x,y]), array([x,y])],[]]
    '''
    while True:
        num_non_empty = int(batch_size * 0.8)
        num_empty = batch_size - num_non_empty
        
        sample_num = 0
        samples = []
        neuron_count = []
        
        while sample_num < num_non_empty:
            current_index = np.random.randint(len(image_arr))
            
            current_image = image_arr[current_index].reshape(512, 512)
            current_locations = locations[current_index]
            
            current_samples, current_count = crop_count_neural(current_image, current_locations, crop_size=crop_size-1)
            
            samples.extend(current_samples)
            neuron_count.extend(current_count)
            sample_num = len(samples)
            
        samples = samples[:num_non_empty]
        neuron_count = neuron_count[:num_non_empty]
        
        non_neuron = 0
        while non_neuron<num_empty:
            current_index = np.random.randint(len(no_neuron_image))
            if len(no_locations[current_index]) == 0:
                current_image = no_neuron_image[current_index].reshape(512, 512)
                if train:
                    one_sample = current_image[50:50+crop_size, 50:50+crop_size]
                else:
                    one_sample = current_image[200:200+crop_size, 200:200+crop_size]
                samples.append(one_sample)
                neuron_count.append(0)
                non_neuron += 1
                
        combined = list(zip(samples, neuron_count))
        random.shuffle(combined)

        samples[:], neuron_count[:] = zip(*combined)
           
        yield 0, (torch.FloatTensor(np.stack(samples).reshape(batch_size, 1, crop_size,crop_size)), (torch.from_numpy(np.array(neuron_count))).float())



def crop_count_neural(image_frame, bbox_center_location, crop_size):
    '''
    @input: 
        image_frame: list of 2-d arrary
        bbox_center_location: list of list with bounding box center position
        crop_size: user initialized, depends on image frame size. 
                    e.g.: if we want a size of 5*5 crop, to set up crop_size = 4
    @output:
        crop_list: list of 2d array, each 2d array represent a crop
        crop_neual_cnt_list: list of interges, each int represent the number of neurals in corresponding crop
    
    '''
    
    def find_bounding_loc(center_index, crop_size = crop_size, image_frame = image_frame):
        '''
        @input: center_index, list of location
        @return: bounding_index, list-of-list [left_bot,left_top,right_bot,right_top]
        @logic: 
            1. given center_index location, return four corner location 
            2. truncate if boundary location out of image
        '''
        tmp = int(crop_size/2)

        left_bot_x = (center_index[0]- tmp) if (center_index[0]- tmp) >= 0 else 0 
        left_top_x = (center_index[0]- tmp) if (center_index[0]- tmp) >= 0 else 0 

        left_bot_y = (center_index[1]- tmp) if (center_index[1]- tmp) >= 0 else 0
        right_bot_y = (center_index[1]- tmp) if (center_index[1]- tmp) >= 0 else 0

        right_bot_x = (center_index[0]+ tmp) if (center_index[0]+ tmp) <= (image_frame.shape[0]-1) else (image_frame.shape[0]-1)
        right_top_x = (center_index[0]+ tmp) if (center_index[0]+ tmp) <= (image_frame.shape[0]-1) else (image_frame.shape[0]-1)

        left_top_y = (center_index[1]+ tmp) if (center_index[1]+ tmp) <= (image_frame.shape[1]-1) else (image_frame.shape[1]-1)  
        right_top_y = (center_index[1]+ tmp) if (center_index[1]+ tmp) <= (image_frame.shape[1]-1) else (image_frame.shape[1]-1)


        left_bot = [left_bot_x, left_bot_y]
        left_top = [left_top_x, left_top_y]
        right_bot = [right_bot_x, right_bot_y]
        right_top = [right_top_x, right_top_y]

        bounding_loc = [left_bot,left_top,right_bot,right_top]

        bounding_loc = [tuple(l) for l in bounding_loc]

        return bounding_loc

    def count_neural(location, bbox_center_location):
        '''
        count how many neurals in a bounding location
        '''
        neural_cnt = 0
        for bbox in bbox_center_location:
            if bbox[0] >= location[0][0] and bbox[0] <= location[2][0] and bbox[1] >= location[0][1] and bbox[1] <= location[1][1]:
                neural_cnt += 1
        return neural_cnt


    bbox_center_location = [tuple(l) for l in bbox_center_location]
    crop_loc_list = list(map(find_bounding_loc, bbox_center_location))
    crop_dict = dict(zip(bbox_center_location, crop_loc_list))
    
    crop_neural_dict = {} 
    for crop_key in crop_dict.keys():
        loction = crop_dict[crop_key]
        neural_cnt = count_neural(loction, bbox_center_location)
        crop_neural_dict[crop_key] = neural_cnt

    # deal with truncated crop, which is out of orginal image frame
    crop_blank_frame = np.zeros((crop_size+1)**2).reshape(crop_size+1,crop_size+1)
    #crop_blank_frame = crop_blank_frame.astype(int)

    crop_list = []
    crop_neual_cnt_list = []
    for crop_key in crop_dict.keys():

        crop = image_frame[crop_dict[crop_key][0][0]:crop_dict[crop_key][2][0]+1, crop_dict[crop_key][0][1]:crop_dict[crop_key][1][1]+1]
        if crop.shape != crop_blank_frame.shape:
            tmp = crop_blank_frame.copy()
            tmp[0:crop.shape[0], 0:crop.shape[1]] = crop
            crop = tmp

        crop_neual_cnt = crop_neural_dict[crop_key]
        crop_list.append(crop)
        crop_neual_cnt_list.append(crop_neual_cnt)
    
    
    return (crop_list,crop_neual_cnt_list)
