import torch
import itertools
import torch.nn as nn
from torch.autograd import Function, Variable


def compute_partial_repr(input_points, control_points):
    N = input_points.size(0)
    
    M = control_points.size(0)
    
    pairwise_diff = input_points.view(N, 1, 2) - control_points.view(1, M, 2)
    
    pairwise_diff_square = pairwise_diff * pairwise_diff
    
    pairwise_dist = pairwise_diff_square[:, :, 0] + pairwise_diff_square[:, :, 1]
   
    repr_matrix = 0.5 * pairwise_dist * torch.log(pairwise_dist)
   
    mask = repr_matrix != repr_matrix
   
    repr_matrix.masked_fill_(mask, 0)
   
    return repr_matrix

class TPSGridGen(nn.Module):

    def __init__(self, target_height, target_width, target_control_points):
        super(TPSGridGen, self).__init__()
        
        assert target_control_points.ndimension() == 2
        assert target_control_points.size(1) == 2
        
        N = target_control_points.size(0)
        
        self.num_points = N
       
        target_control_points = target_control_points.float()
        

        forward_kernel = torch.zeros(N + 3, N + 3)
        
        target_control_partial_repr = compute_partial_repr(target_control_points, target_control_points)
       
        forward_kernel[:N, :N].copy_(target_control_partial_repr)
        
        forward_kernel[:N, -3].fill_(1)
       
        forward_kernel[-3, :N].fill_(1)
       
        forward_kernel[:N, -2:].copy_(target_control_points)
      
        forward_kernel[-2:, :N].copy_(target_control_points.transpose(0, 1))
       
        inverse_kernel = torch.inverse(forward_kernel)
        

        # create target cordinate matrix
        HW = target_height * target_width
       
        target_coordinate = list(itertools.product(range(target_height), range(target_width)))
       
        target_coordinate = torch.Tensor(target_coordinate) # HW x 2
       
        Y, X = target_coordinate.split(1, dim = 1)
        Y = Y * 2 / (target_height - 1) - 1
        X = X * 2 / (target_width - 1) - 1
       
        target_coordinate = torch.cat([X, Y], dim = 1) # convert from (y, x) to (x, y)
       
        target_coordinate_partial_repr = compute_partial_repr(target_coordinate, target_control_points)
       
        target_coordinate_repr = torch.cat([
            target_coordinate_partial_repr, torch.ones(HW, 1), target_coordinate
        ], dim = 1)
       
        self.inverse_kernel = inverse_kernel
        
        self.padding_matrix = torch.zeros(3, 2)
        
        self.target_coordinate_repr = target_coordinate_repr

    def forward(self, source_control_points):
        assert source_control_points.ndimension() == 3
        assert source_control_points.size(1) == self.num_points
        assert source_control_points.size(2) == 2
        batch_size = source_control_points.size(0)
       

        Y = torch.cat([source_control_points, Variable(self.padding_matrix.expand(batch_size, 3, 2))], 1)
       
        mapping_matrix = torch.matmul(Variable(self.inverse_kernel), Y)
       
        source_coordinate = torch.matmul(Variable(self.target_coordinate_repr), mapping_matrix)
       
        return source_coordinate
