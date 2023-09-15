import torch
import torch.nn.functional as F
import numpy as np
import math


class NCC:
    def __init__(self, win=9):
        self.win = win

    def loss(self, y_true, y_pred):

        Ii = y_true
        Ji = y_pred

        # get dimension of volume
        ndims = len(list(Ii.size())) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims
        channel_num = Ii.shape[1]

        # set window size
        win = [self.win] * ndims

        # compute filters
        sum_filt = torch.ones([channel_num, 1, *win]).to(y_pred.device)

        pad_no = math.floor(win[0] / 2)

        if ndims == 1:
            stride = (1)
            padding = (pad_no)
        elif ndims == 2:
            stride = (1, 1)
            padding = (pad_no, pad_no)
        else:
            stride = (1, 1, 1)
            padding = (pad_no, pad_no, pad_no)

        # get convolution function
        conv_fn = getattr(F, 'conv%dd' % ndims)

        # compute CC squares
        I2 = Ii * Ii
        J2 = Ji * Ji
        IJ = Ii * Ji

        I_sum = conv_fn(Ii, sum_filt, stride=stride, padding=padding, groups=channel_num)
        J_sum = conv_fn(Ji, sum_filt, stride=stride, padding=padding, groups=channel_num)
        I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding, groups=channel_num)
        J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding, groups=channel_num)
        IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding, groups=channel_num)

        win_size = np.prod(win)
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        cc = cross * cross / (I_var * J_var + 1e-5)

        return -torch.mean(cc)

    
class Grad:
    def __init__(self, penalty='l1', loss_mult=None):
        self.penalty = penalty
        self.loss_mult = loss_mult

    def loss(self, _, y_pred):
        
        dy = torch.abs(y_pred[:, :, 1:, :, :] - y_pred[:, :, :-1, :, :])
        dx = torch.abs(y_pred[:, :, :, 1:, :] - y_pred[:, :, :, :-1, :])
        dz = torch.abs(y_pred[:, :, :, :, 1:] - y_pred[:, :, :, :, :-1])

        if self.penalty == 'l2':
            dy = dy * dy
            dx = dx * dx
            dz = dz * dz

        d = torch.mean(dx) + torch.mean(dy) + torch.mean(dz)
        grad = d / 3.0

        if self.loss_mult is not None:
            grad *= self.loss_mult
        
        return grad
    

# This NJD loss works at PyTorch=1.10 but failed at PyTorch=1.13 for unknown reasons
class NJD:
    def __init__(self, Lambda=1e-5):
        self.Lambda = Lambda
        
    def get_Ja(self, displacement):

        D_y = (displacement[:,1:,:-1,:-1,:] - displacement[:,:-1,:-1,:-1,:])
        D_x = (displacement[:,:-1,1:,:-1,:] - displacement[:,:-1,:-1,:-1,:])
        D_z = (displacement[:,:-1,:-1,1:,:] - displacement[:,:-1,:-1,:-1,:])

        D1 = (D_x[...,0]+1)*( (D_y[...,1]+1)*(D_z[...,2]+1) - D_z[...,1]*D_y[...,2])
        D2 = (D_x[...,1])*(D_y[...,0]*(D_z[...,2]+1) - D_y[...,2]*D_x[...,0])
        D3 = (D_x[...,2])*(D_y[...,0]*D_z[...,1] - (D_y[...,1]+1)*D_z[...,0])
        
        return D1-D2+D3

    def loss(self, _, y_pred):

        displacement = y_pred.permute(0, 2, 3, 4, 1)
        Ja = self.get_Ja(displacement)
        Neg_Jac = 0.5*(torch.abs(Ja) - Ja)
    
        return self.Lambda*torch.sum(Neg_Jac)
    
    
class KL():
    def __init__(self, prior_lambda=100):
        self.prior_lambda = prior_lambda

    def _adj_filt(self, ndims):

        # inner filter 3x3x3
        filt_inner = np.zeros([3] * ndims)
        for j in range(ndims):
            o = [[1]] * ndims
            o[j] = [0, 2]
            filt_inner[np.ix_(*o)] = 1

        # full filter, that makes sure the inner filter is applied 
        filt = np.zeros([ndims, 1] + [3] * ndims)
        for i in range(ndims):
            filt[i,0] = filt_inner
        
        return torch.Tensor(filt)

    def _degree_matrix(self, vol_shape):
        
        ndims = len(vol_shape)
        x = torch.ones([1, ndims, *vol_shape])
        filt = self._adj_filt(ndims)
        
        conv_fn = getattr(F, 'conv%dd' % ndims)
        return conv_fn(x, filt, padding='same', groups=ndims)

    def prec_loss(self, y_pred):

        vol_shape = y_pred.shape[2:]
        ndims = len(vol_shape)
        
        sm = 0
        for i in range(ndims):
            d = i + 2
            # permute dimensions to put the ith dimension first
            r = [d, *range(d), *range(d + 1, ndims + 2)]
            y = y_pred.permute(r)
            df = y[1:] - y[:-1]
            sm += torch.mean(df * df)
        
        return 0.5 * sm / ndims

    def loss(self, _, y_pred):
        """
        KL loss
        y_pred is assumed to be D*2 channels: first D for mean, next D for logsigma
        D (number of dimensions) should be 1, 2 or 3
        """

        # prepare inputs
        vol_shape = y_pred.shape[2:]
        ndims = len(vol_shape)
        mean = y_pred[:,0:ndims]
        log_sigma = y_pred[:,ndims:]

        # compute the degree matrix
        D = self._degree_matrix(vol_shape).to(y_pred.device)

        # sigma terms
        sigma_term = self.prior_lambda * D * torch.exp(log_sigma) - log_sigma
        sigma_term = torch.mean(sigma_term)

        # precision terms
        prec_term = self.prior_lambda * self.prec_loss(mean)

        return 0.5 * ndims * (sigma_term + prec_term)
