# py imports
import os
import sys
import glob
import time
import numpy as np
import torch
import scipy.ndimage
from argparse import ArgumentParser

# project imports
import networks
import datagenerators


def Dice(vol1, vol2, labels=None, nargout=1):
    
    if labels is None:
        labels = np.unique(np.concatenate((vol1, vol2)))
        labels = np.delete(labels, np.where(labels == 0))  # remove background

    if len(labels)==0:
        return 0.0
        
    dicem = np.zeros(len(labels))
    for idx, lab in enumerate(labels):
        vol1l = vol1 == lab
        vol2l = vol2 == lab
        top = 2 * np.sum(np.logical_and(vol1l, vol2l))
        bottom = np.sum(vol1l) + np.sum(vol2l)
        bottom = np.maximum(bottom, np.finfo(float).eps)  # add epsilon.
        dicem[idx] = top / bottom

    if nargout == 1:
        return dicem
    else:
        return (dicem, labels)


def NJD(displacement):

    D_y = (displacement[1:,:-1,:-1,:] - displacement[:-1,:-1,:-1,:])
    D_x = (displacement[:-1,1:,:-1,:] - displacement[:-1,:-1,:-1,:])
    D_z = (displacement[:-1,:-1,1:,:] - displacement[:-1,:-1,:-1,:])

    D1 = (D_x[...,0]+1)*( (D_y[...,1]+1)*(D_z[...,2]+1) - D_z[...,1]*D_y[...,2])
    D2 = (D_x[...,1])*(D_y[...,0]*(D_z[...,2]+1) - D_y[...,2]*D_x[...,0])
    D3 = (D_x[...,2])*(D_y[...,0]*D_z[...,1] - (D_y[...,1]+1)*D_z[...,0])
    Ja_value = D1-D2+D3
    
    return np.sum(Ja_value<0)


def test(test_dir,
         test_pairs,
         device, 
         load_model):
    
    # preparation
    test_pairs = np.load(test_dir+test_pairs, allow_pickle=True)

    # device handling
    if 'gpu' in device:
        os.environ['CUDA_VISIBLE_DEVICES'] = device[-1]
        device = 'cuda'
        torch.backends.cudnn.deterministic = True
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        device = 'cpu'
    
    # prepare model
    model = networks.AutoFuse(for_train=False)
    print('loading', load_model)
    state_dict = torch.load(load_model, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # transfer model
    SpatialTransformer = networks.SpatialTransformer_block(mode='nearest')
    SpatialTransformer.to(device)
    SpatialTransformer.eval()
    
    # testing loop
    Dice_result = [] 
    NJD_result = []
    Seg_result = []
    Runtime_result = []
    for test_pair in test_pairs:
        print(test_pair)
        
        # generate inputs (and true outputs)
        fix_vol, fix_seg = datagenerators.load_by_name(test_dir, test_pair[0])
        mov_vol, mov_seg = datagenerators.load_by_name(test_dir, test_pair[1])
            
        # convert input to tensors
        fix_vol = torch.from_numpy(fix_vol).to(device).float()
        fix_seg = torch.from_numpy(fix_seg).to(device).float()
        mov_vol = torch.from_numpy(mov_vol).to(device).float()
        mov_seg = torch.from_numpy(mov_seg).to(device).float()
        
        t = time.time()
        with torch.no_grad():
            pred = model(fix_vol, mov_vol, torch.zeros(1).to(device)) # feed zero tensor as pseudo input
        Runtime_val = time.time() - t
        
        warp_seg = SpatialTransformer(mov_seg, pred[0])
        warp_seg = warp_seg.detach().cpu().numpy().squeeze()
        fix_seg = fix_seg.detach().cpu().numpy().squeeze()
        mov_seg = mov_seg.detach().cpu().numpy().squeeze()
        
        Dice_val = Dice(warp_seg, fix_seg)
        Dice_result.append(Dice_val)
        
        flow = pred[0].detach().cpu().permute(0, 2, 3, 4, 1).numpy().squeeze()
        NJD_val = NJD(flow)
        NJD_result.append(NJD_val)
        
        fix_pred = np.argmax(pred[1].detach().cpu().numpy().squeeze(), axis=0)
        Dice_fix = Dice(fix_pred, fix_seg)
        Seg_result.append(Dice_fix)
            
        mov_pred = np.argmax(pred[2].detach().cpu().numpy().squeeze(), axis=0)
        Dice_mov = Dice(mov_pred, mov_seg)
        Seg_result.append(Dice_mov)
        
        Runtime_result.append(Runtime_val)
        
        print('Dice: {:.3f} ({:.3f})'.format(np.mean(Dice_val), np.std(Dice_val)))
        print('NJD: {:.3f}'.format(NJD_val))
        print('Seg Dice: {:.3f}'.format(np.mean(Dice_fix)))
        print('Runtime: {:.3f}'.format(Runtime_val))

    Dice_result = np.array(Dice_result)
    print('Average Dice: {:.3f} ({:.3f})'.format(np.mean(Dice_result), np.std(Dice_result)))
    NJD_result = np.array(NJD_result)
    print('Average NJD: {:.3f} ({:.3f})'.format(np.mean(NJD_result), np.std(NJD_result)))
    Seg_result = np.array(Seg_result)
    print('Average Seg Dice: {:.3f} ({:.3f})'.format(np.mean(Seg_result), np.std(Seg_result)))
    Runtime_result = np.array(Runtime_result)
    print('Average Runtime mean: {:.3f} ({:.3f})'.format(np.mean(Runtime_result[1:]), np.std(Runtime_result[1:])))


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--test_dir", type=str,
                        dest="test_dir", default='./',
                        help="test folder")
    parser.add_argument("--test_pairs", type=str,
                        dest="test_pairs", default='test_pairs.npy',
                        help="testing pairs(.npy)")
    parser.add_argument("--device", type=str, default='gpu0',
                        dest="device", help="cpu or gpuN")
    parser.add_argument("--load_model", type=str,
                        dest="load_model", default='./',
                        help="load model file to initialize with")

    args = parser.parse_args()
    test(**vars(args))
