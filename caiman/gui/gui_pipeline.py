""" pipeline for showing the GUI function in CaImAn


See Also
------------


"""
# \package None
# \version   1.0
# \copyright GNU General Public License v2.0
# \date Created on june 2017
# \author: Jeremie KALFON


from __future__ import division
from __future__ import print_function
from builtins import str
import matplotlib

matplotlib.use('agg')
from caiman.utils.utils import download_demo
import cv2
import glob

try:
    cv2.setNumThreads(1)
except:
    print('Open CV is naturally single threaded')

try:
    if __IPYTHON__:
        print((1))
        # this is used for debugging purposes only. allows to reload classes
        # when changed
        get_ipython().magic('load_ext autoreload')
        get_ipython().magic('autoreload 2')
except NameError:
    print('Not IPYTHON')
    pass
import caiman as cm
import numpy as np
import os
import time
import copy
from caiman.utils.visualization import plot_contours
from caiman.source_extraction.cnmf import cnmf as cnmf
from caiman.motion_correction import MotionCorrect
from caiman.components_evaluation import estimate_components_quality

import caiman.gui as gui

# GLOBAL VAR
params_movie = {'fname': ['DemoMovieJ.tif'],
                'niter_rig': 1,
                'max_shifts': (20, 20),  # maximum allow rigid shift
                'splits_rig': 20,  # for parallelization split the movies in  num_splits chuncks across time
                # if none all the splits are processed and the movie is saved
                'num_splits_to_process_rig': None,

                # intervals at which patches are laid out for motion correction
                'merge_thresh': 0.8,  # merging threshold, max correlation allowed
                'rf': 15,  # half-size of
                # the patches in pixels. rf=25, patches are 50x50
                'stride_cnmf': 6,  # amount of overlap between the patches in pixels
                'K': 4,  # number of components per patch
                'p': 1,  # order of the autoregressive system

                # if dendritic. In this case you need to set init_method to sparse_nmf
                'is_dendrites': False,
                'init_method': 'greedy_roi',
                'gSig': [4, 4],  # expected half size of neurons
                'alpha_snmf': None,  # this controls sparsity

                'final_frate': 30,
                'r_values_min_patch': .8,  # threshold on space consistency
                'fitness_min_patch': -40,  # threshold on time variability
                'fitness_delta_min_patch': -40,  # threshold on time variability of the diff of the activity
                'r_values_min_full': .85,
                'fitness_min_full': - 50,
                'fitness_delta_min_full': - 50,

                'Npeaks': 10,
                'only_init_patch': True,
                'gnb': 1,
                'memory_fact': 1,
                'n_chunks': 10
                }
params_display = {
    'downsample_ratio': .2,
    'thr_plot': 0.9
}


#%%
fname = params_movie['fname']
niter_rig = params_movie['niter_rig']
max_shifts = params_movie['max_shifts']
splits_rig = params_movie['splits_rig']
num_splits_to_process_rig = params_movie['num_splits_to_process_rig']

download_demo(fname[0])
fname = os.path.join('example_movies', fname[0])
m_orig = cm.load(fname)
min_mov = m_orig[:400].min()
#%%
################ RIG CORRECTION #################
mc = MotionCorrect(fname, min_mov,
                   max_shifts=max_shifts, niter_rig=niter_rig
                   , splits_rig=splits_rig,
                   num_splits_to_process_rig=num_splits_to_process_rig,
                   shifts_opencv=True, nonneg_movie=True)
mc.motion_correct_rigid(save_movie=True)
m_rig = cm.load(mc.fname_tot_rig)
bord_px_rig = np.ceil(np.max(mc.shifts_rig)).astype(np.int)
###########################################
#%%

if not params_movie.has_key('max_shifts'):
    fnames = params_movie['fname']
    border_to_0 = 0
else:
    fnames = [mc.fname_tot_rig]
    border_to_0 = bord_px_rig
    m_els = m_rig

idx_xy = None
add_to_movie = -np.nanmin(m_els) + 1  # movie must be positive
remove_init = 0
downsample_factor = 1
base_name = fname[0].split('/')[-1][:-4]
name_new = cm.save_memmap_each(fnames,base_name=base_name, resize_fact=(
    1, 1, downsample_factor), remove_init=remove_init,
                               idx_xy=idx_xy, add_to_movie=add_to_movie, border_to_0=border_to_0)
name_new.sort()

if len(name_new) > 1:
    fname_new = cm.save_memmap_join(
        name_new, base_name='Yr', n_chunks=params_movie['n_chunks'])
else:
    print('One file only, not saving!')
    fname_new = name_new[0]

Yr, dims, T = cm.load_memmap(fname_new)
print("##################################")
print(dims)
images = np.reshape(Yr.T, [T] + list(dims), order='F')
Y = np.reshape(Yr, dims + (T,), order='F')

if np.min(images) < 0:
    # TODO: should do this in an automatic fashion with a while loop at the 367 line
    raise Exception('Movie too negative, add_to_movie should be larger')
if np.sum(np.isnan(images)) > 0:
    # TODO: same here
    raise Exception('Movie contains nan! You did not remove enough borders')

Cn = cm.local_correlations(Y)
Cn[np.isnan(Cn)] = 0
p = params_movie['p']
merge_thresh = params_movie['merge_thresh']
rf = params_movie['rf']
stride_cnmf = params_movie['stride_cnmf']
K = params_movie['K']
init_method = params_movie['init_method']
gSig = params_movie['gSig']
alpha_snmf = params_movie['alpha_snmf']

if params_movie['is_dendrites'] == True:
    if params_movie['init_method'] is not 'sparse_nmf':
        raise Exception('dendritic requires sparse_nmf')
    if params_movie['alpha_snmf'] is None:
        raise Exception('need to set a value for alpha_snmf')

#%%
################ CNMF PART PATCH #################
cnm = cnmf.CNMF(n_processes=1, k=K,gSig=gSig, merge_thresh=params_movie['merge_thresh'], p=params_movie['p'],
                 rf=rf, stride=stride_cnmf, memory_fact=params_movie['memory_fact'],
                method_init=init_method, alpha_snmf=alpha_snmf, only_init_patch=params_movie['only_init_patch'],
                gnb=params_movie['gnb'], method_deconvolution='oasis')
cnm = cnm.fit(images)
A_tot = cnm.A
C_tot = cnm.C
YrA_tot = cnm.YrA
b_tot = cnm.b
f_tot = cnm.f
#%%
######### DISCARDING
print(('Number of components:' + str(A_tot.shape[-1])))
final_frate = params_movie['final_frate']
r_values_min = params_movie['r_values_min_patch']  # threshold on space consistency
fitness_min = params_movie['fitness_delta_min_patch']  # threshold on time variability
fitness_delta_min = params_movie['fitness_delta_min_patch']
Npeaks = params_movie['Npeaks']
traces = C_tot + YrA_tot
idx_components, idx_components_bad = estimate_components_quality(
    traces, Y, A_tot, C_tot, b_tot, f_tot, final_frate=final_frate,
    Npeaks=Npeaks, r_values_min=r_values_min, fitness_min=fitness_min,
    fitness_delta_min=fitness_delta_min)
#######
A_tot = A_tot.tocsc()[:, idx_components]
C_tot = C_tot[idx_components]
#################### ########################
#%%
################ CNMF PART FULL #################
cnm = cnmf.CNMF(n_processes=1, k=A_tot.shape, gSig=gSig, merge_thresh=merge_thresh, p=p, Ain=A_tot, Cin=C_tot,
                f_in=f_tot, rf=None, stride=None, method_deconvolution='oasis')
cnm = cnm.fit(images)
#%%
A, C, b, f, YrA, sn = cnm.A, cnm.C, cnm.b, cnm.f, cnm.YrA, cnm.sn


A_thr = cm.source_extraction.cnmf.spatial.threshold_components(
    A.toarray(), dims, medw=None, thr_method='max', maxthr=0.2, nrgthr=0.99, extract_cc=True,
    se=None, ss=None)

A_thr = A_thr > 0
C_thr = C
n_frames_per_bin = 10

C_thr = np.array([CC.reshape([-1, n_frames_per_bin]).max(1) for CC in C_thr])
maskROI = A_thr[:, :].reshape([dims[0], dims[1], -1], order='F').transpose([2, 0, 1]) * 1.
gui.gui.gui.GUI(mov = m_rig,roi = maskROI, traces=C_thr.T)



final_frate = params_movie['final_frate']
r_values_min = params_movie['r_values_min_full']  # threshold on space consistency
fitness_min = params_movie['fitness_delta_min_full']  # threshold on time variability
fitness_delta_min = params_movie['fitness_delta_min_full']
Npeaks = params_movie['Npeaks']

#%%
############ DISCARDING
traces = C + YrA
idx_components, idx_components_bad, fitness_raw, fitness_delta, r_values = estimate_components_quality(
    traces, Y, A, C, b, f, final_frate=final_frate, Npeaks=Npeaks, r_values_min=r_values_min,
    fitness_min=fitness_min,
    fitness_delta_min=fitness_delta_min, return_all=True)
#%%
##########
A_tot_full = A_tot.tocsc()[:, idx_components]
C_tot_full = C_tot[idx_components]
#%%
#################### ########################
log_files = glob.glob('*_LOG_*')
for log_file in log_files:
    os.remove(log_file)
    ############ assertions ##################