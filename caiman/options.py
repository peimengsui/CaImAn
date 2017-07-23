# -*- coding: utf-8 -*-
""" A set of utilities, mostly for post-processing and visualization

We put arrays on disk as raw bytes, extending along the first dimension.
Alongside each array x we ensure the value x.dtype which stores the string
description of the array's dtype.

See Also:
------------

@url
.. image::
@author  epnev
"""
#\package caiman/dource_ectraction/cnmf
#\version   1.0
#\copyright GNU General Public License v2.0
#\date Created on Sat Sep 12 15:52:53 2015

from __future__ import division
from __future__ import print_function

import numpy as np
import caiman as cm
import psutil

#%%
def Params(name = 'demoMovieJ', nbneuron=30, neuronsize=None,remove_very_bad_comps = True,
                automode=True, power=False):
    """Dictionary for setting the CNMF parameters.

    Any parameter that is not set get a default value specified
    by the dictionary default options

    PRE-PROCESS PARAMS#############
    sn: None,
        noise level for each pixel

    noise_range: [0.25, 0.5]
             range of normalized frequencies over which to average

    noise_method': 'mean'
             averaging method ('mean','median','logmexp')

    max_num_samples_fft': 3*1024

    n_pixels_per_process: 1000

    compute_g': False
        flag for estimating global time constant

    p : 2
         order of AR indicator dynamics

    lags: 5
        number of autocovariance lags to be considered for time constant estimation

    include_noise: False
            flag for using noise values when estimating g

    pixels: None
         pixels to be excluded due to saturation

    check_nan: True

    INIT PARAMS###############

    K:     30
        number of components

    gSig: [5, 5]
          size of bounding box

    gSiz: [int(round((x * 2) + 1)) for x in gSig],

    ssub:   2
        spatial downsampling factor

    tsub:   2
        temporal downsampling factor

    nIter: 5
        number of refinement iterations

    kernel: None
        user specified template for greedyROI

    maxIter: 5
        number of HALS iterations

    method: method_init
        can be greedy_roi or sparse_nmf, local_NMF

    max_iter_snmf : 500

    alpha_snmf: 10e2

    sigma_smooth_snmf : (.5,.5,.5)

    perc_baseline_snmf: 20

    nb:  1
        number of background components

    normalize_init:
        whether to pixelwise equalize the movies during initialization

    options_local_NMF:
        dictionary with parameters to pass to local_NMF initializaer

    SPATIAL PARAMS##########

        dims: dims
            number of rows, columns [and depths]

        method: 'dilate','ellipse', 'dilate'
            method for determining footprint of spatial components ('ellipse' or 'dilate')

        dist: 3
            expansion factor of ellipse
        n_pixels_per_process: n_pixels_per_process
            number of pixels to be processed by eacg worker

        medw: (3,)*len(dims)
            window of median filter
        thr_method: 'nrg'
           Method of thresholding ('max' or 'nrg')

        maxthr: 0.1
            Max threshold

        nrgthr: 0.9999
            Energy threshold


        extract_cc: True
            Flag to extract connected components (might want to turn to False for dendritic imaging)

        se: np.ones((3,)*len(dims), dtype=np.uint8)
             Morphological closing structuring element

        ss: np.ones((3,)*len(dims), dtype=np.uint8)
            Binary element for determining connectivity

         nb

        method_ls:'lasso_lars'
            'nnls_L0'. Nonnegative least square with L0 penalty
            'lasso_lars' lasso lars function from scikit learn
            'lasso_lars_old' lasso lars from old implementation, will be deprecated

        TEMPORAL PARAMS###########

        ITER: 2
            block coordinate descent iterations

        method:'oasis', 'cvxpy',  'oasis'
            method for solving the constrained deconvolution problem ('oasis','cvx' or 'cvxpy')
            if method cvxpy, primary and secondary (if problem unfeasible for approx solution)

        solvers: ['ECOS', 'SCS']
             solvers to be used with cvxpy, can be 'ECOS','SCS' or 'CVXOPT'

        p:
            order of AR indicator dynamics

        memory_efficient: False

        bas_nonneg: True
            flag for setting non-negative baseline (otherwise b >= min(y))

        noise_range: [.25, .5]
            range of normalized frequencies over which to average

        noise_method: 'mean'
            averaging method ('mean','median','logmexp')

        lags: 5,
            number of autocovariance lags to be considered for time constant estimation

        fudge_factor: .96
            bias correction factor (between 0 and 1, close to 1)

        nb

        verbosity: False

        block_size : block_size
            number of pixels to process at the same time for dot product. Make it smaller if memory problems
    """
    options = dict()

    # find the dims fo Y
    if type(Y) is tuple:
        dims, T = Y[:-1], Y[-1]
    else:
        dims, T = Y.shape[:-1], Y.shape[-1]
    c, dview, n_processes = cm.cluster.setup_cluster(
        backend='local', n_processes=None, single_thread=False)
    gSig = [neuronsize, neuronsize] if neuronsize is not None else [-1, -1]
######### OPTIONS IF NOT AUTO MODE ####################### SEE INFO BELLOW ###########

    alpha_snmf = 10e2
    check_nan = True
    fitpatch = -40
    fitfull = -50
    frate = 10
    gnb =1
    initonly = True
    lags = 5
    max_shifts=[1,1]
    maxdevrig = 1
    memory_efficient = False
    merge_thresh = 0.8
    method_init = 'greedy_roi'
    nb = 1
    noise_range = [0.25, 0.5]
    normalize_init = True
    options_local_NMF = None
    overlaps = 12
    p = 1
    p_ssub = 2
    p_tsub = 2
    rf = 20
    splits_rig = 4
    splits = 14
    strides = 48
    ssub = 2
    thr = 0.8
    tsub = 2
    upfact = 3

    reference_mem = 10e9
    mem_per_pix = 3.6977678498329843e-09
#################### COMPUTING ################

    if automode :
        used = np.array(psutil.virtual_memory()[2])
        memory = np.array(psutil.virtual_memory()[0])  ## we can refer to this one more
        avail_memory = np.array(psutil.virtual_memory()[1]) ##it should change drastically depending on the process

        cpu = np.array(psutil.cpu_freq()[2])
        mem = int(memory / 1e9)
        cpu = int(cpu / 1e3)
        pwr = n_processes * cpu
        if (used > 80 )or(avail_memory/memory)<0.3:
            print("you may need to close some apps and to kill some processes to make CaImaN run faster")

        filesize
        n_chunks = int((filesize*n_processes)/(mem-4)+1)
        if power:
            if pwr-10 >power:
                print("your computer does not seem to have that kind ")


        #TODO check is this a good way ?
        avail_memory_per_process = avail_memory / 2. ** 30 / n_processes
        n_pixels_per_process = np.int(avail_memory_per_process / 8. / mem_per_pix / T)
        n_pixels_per_process = np.int(np.minimum(n_pixels_per_process, np.prod(dims) // n_processes))
        block_size = n_pixels_per_process

        #find gsig



        #find p



        max_shifts = mxshifts(dims)

        #compute strides overlaps with virtual memory and number of processes

        #rf K  gsig, size =  image quality, n process, power, approx nb of neuron




    options['params_display'] = {
        'downsample_ratio': .2,
        'thr_plot': 0.9
    }
######################################### MORE COMPLEX OPTIONS ################################ 34 more to choose here #
    options['params_movie'] = {
           'fname': [name+'.tif'],
           'max_shifts': max_shifts, # maximum allow rigid shift (2,2)
           'niter_rig': 1,              ####### number of iteration of rigid motion correction
           'splits_rig': splits_rig,  # for parallelization split the movies in  num_splits chuncks across time #@inferrable
           'num_splits_to_process_rig': None,  # if none all the splits are processed and the movie is saved #@inferrable
           'strides': strides,  # intervals at which patches are laid out for motion correction #@inferrable???
           'overlaps': overlaps,  # overlap between pathes (size of patch strides+overlaps)  #@inferrable???
           'splits_els': splits,  # for parallelization split the movies in  num_splits chuncks across time  #@inferrable
           'num_splits_to_process_els': [splits, None],
           # if none all the splits are processed and the movie is saved
           'upsample_factor_grid': upfact,  # upsample factor to avoid smearing when merging patches  #@inferrable
           'max_deviation_rigid': maxdevrig,  # maximum deviation allowed for patch with respect to rigid shift  #@inferrable
           'p': p,  # order of the autoregressive system
           'merge_thresh': merge_thresh,  # merging threshold, max correlation allow
           'rf': rf,  # half-size of the patches in pixels. rf=25, patches are 50x50    20  #@inferrable
           'stride_cnmf': (gSig[0]+gSig[1])/2,  ############# amounpl.it of overlap between the patches in pixels
           'K': nbneuron,  # number of components per patch
           'is_dendrites': False,  # if dendritic. In this case you need to set init_method to sparse_nmf
           'init_method': method_init,
           'gSig': gSig,  # expected half size of neurons
           'alpha_snmf': alpha_snmf,  # this controls sparsity
           'final_frate': frate,
           'r_values_min_patch': .7,  ################ threshold on space consistency
           'fitness_min_patch': fitpatch,  # threshold on time variability
           # threshold on derivative of time variability (if nonsparse activity)
           'fitness_delta_min_patch': fitpatch,
           'Npeaks': 10, ###################
           'r_values_min_full': .85, #################
           'fitness_min_full': fitfull,
           'fitness_delta_min_full': fitfull,
           'only_init_patch': initonly ,
           'gnb': gnb, # number of background components
           'n_chunks': n_chunks ###################
                               }

    options['parr']={
        'c':c,
        'dview':dview
    }
    options['init_params'] = {
          'K': K,  # number of components
          'gSig': gSig,  # size of bounding box
          'gSiz': [int(round((x * 2) + 1)) for x in gSig], ####################
          'ssub': ssub,  # spatial downsampling factor
          'tsub': tsub,  # temporal downsampling factor
          'nIter': 5,  ############## number of refinement iterations
          'kernel': None,  ################# user specified template for greedyROI
          'maxIter': 5,  ############ number of HALS iterations
          'method': method_init,  # can be greedy_roi or sparse_nmf, local_NMF
          'max_iter_snmf': 500, ###############
          'alpha_snmf': alpha_snmf,
          'sigma_smooth_snmf': (.5, .5, .5), ################
          'perc_baseline_snmf': 20, #############
          'nb': nb,  # number of background components
          # whether to pixelwise equalize the movies during initialization
          'normalize_init': normalize_init,
          # dictionary with parameters to pass to local_NMF initializaer
          'options_local_NMF': options_local_NMF
          }
    options['preprocess_params'] = {
        'sn': None,  ############## noise level for each pixel
        # range of normalized frequencies over which to average
        'noise_range': noise_range,
        # averaging method ('mean','median','logmexp')
        'noise_method': 'mean',
        'max_num_samples_fft': 3 * 1024, ###################
        'n_pixels_per_process': n_pixels_per_process,
        'compute_g': False,  ################ flag for estimating global time constant
        'p': p,  # order of AR indicator dynamics
        # number of autocovariance lags to be considered for time constant estimation
        'lags': lags,
        'include_noise': False,  ################### flag for using noise values when estimating g
        'pixels': None, #################
        # pixels to be excluded due to saturation
        'check_nan': check_nan
        }
    options['patch_params'] = {
        'ssub': p_ssub,  # spatial downsampling factor
        'tsub': p_tsub,  # temporal downsampling factor
        'only_init': initonly,
        'skip_refinement': False, #####################
        'remove_very_bad_comps': remove_very_bad_comps
    }
    options['merging'] = {
        'thr': thr,
    }
    options['temporal_params'] = {
        'ITER': 2,  ################## block coordinate descent iterations
        # method for solving the constrained deconvolution problem ('oasis','cvx' or 'cvxpy')
        'method': 'oasis',  #################### 'cvxpy', # 'oasis'
                            # if method cvxpy, primary and secondary (if problem unfeasible for approx
                            # solution) solvers to be used with cvxpy, can be 'ECOS','SCS' or 'CVXOPT'
        'solvers': ['ECOS', 'SCS'], ########################
        'p': p,  # order of AR indicator dynamics
        'memory_efficient': memory_efficient,
        # flag for setting non-negative baseline (otherwise b >= min(y))
        'bas_nonneg': True, #######################
        # range of normalized frequencies over which to average
        'noise_range': noise_range,
        'noise_method': 'mean',  ################### averaging method ('mean','median','logmexp')
        'lags': lags,  # number of autocovariance lags to be considered for time constant estimation
        'fudge_factor': .96,  ############### bias correction factor (between 0 and 1, close to 1)
        'nb': gnb,  # number of background components
        'verbosity': False,
        # number of pixels to process at the same time for dot product. Make it smaller if memory problems
        'block_size': block_size
    }
    options['spatial_params'] = {
        'dims': dims,  # number of rows, columns [and depths]
        # method for determining footprint of spatial components ('ellipse' or 'dilate')
        'method': 'dilate',  ################'ellipse', 'dilate',
        'dist': 3,  ################## expansion factor of ellipse
        'n_pixels_per_process': n_pixels_per_process,  # number of pixels to be processed by eacg worker
        'medw': (3,) * len(dims),  ################### window of median filter
        'thr_method': 'nrg',  ############## Method of thresholding ('max' or 'nrg')
        'maxthr': 0.1,  ############ Max threshold
        'nrgthr': 0.9999,  ############# Energy threshold
        # Flag to extract connected components (might want to turn to False for dendritic imaging)
        'extract_cc': True, #####################
        'se': np.ones((3,) * len(dims), dtype=np.uint8),  ################ Morphological closing structuring element
        'ss': np.ones((3,) * len(dims), dtype=np.uint8),  ############## Binary element for determining connectivity
        'nb': gnb,  # number of background components
        'method_ls': 'lasso_lars',  ###########  'nnls_L0'. Nonnegative least square with L0 penalty
    }

    return options
#%%

def mxshifts(dims, intensity=1):
    """
 compute automatically the max shifts

maxshifts are computed this way after reallizing that the rigid motion correction is really robust and
will not behave differently with values that are 7 times higher than what is really required
    """
    X = dims[0]/(10-intensity)
    Y = dims[1]/(10 - intensity)
    return [X,Y]
