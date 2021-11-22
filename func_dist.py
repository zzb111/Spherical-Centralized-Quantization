
from math import isnan
from tool import out_float_tensor
import torch

def get_L2_dist(a,b):
    # return torch.nn.functional.pairwise_distance(a,b,p=2,eps= 1e-9)
    a_64 = a.double()
    b_64 = b.double()
    norm_1 = torch.norm(a_64,dim=1).repeat(b_64.size(0),1).t()
    norm_1 = norm_1 ** 2

    norm_2 = torch.norm(b_64,dim=1).repeat(a_64.size(0),1)
    norm_2 = norm_2 ** 2

    # print(norm_2.dtype)

    # if(  abs( norm_1.mean().item() - 360 ** 2 ) > 0.1 ):
    #     print( 'norm_1 = ',norm_1.mean().item())
    #     exit()
    # if(  abs( norm_2.mean().item() - 360 ** 2 ) > 0.1 ):
    #     print( 'norm_2 = ',norm_2.mean().item())
    #     exit()
    # print('norm_1 = ')
    # out_float_tensor(norm_1)
    # print(norm_1.item())
    # print('norm_2 = ')
    # out_float_tensor(norm_2)
    # print(norm_2.item())

    mul = torch.mm(a_64,b_64.t())
    # print('mul = ')
    # print(mul.item())
    # print(mul)
    # if ( float('nan') in mul ):
    # for i in range(mul.size(0)):
    #     for j in range( mul.size(1) ):
    #         if ( torch.isnan( mul[i][j] ) ):
    #             print('nan error')
                # out_float_tensor( a[i] )
                # out_float_tensor( b[j] )
                # exit()

    c_64 = (norm_1+norm_2-2*mul)
    # c = torch.tensor(c_64)
    # c = (200-2*mul)
    # if ( torch.isnan( c.mean() ) ):
    #     print('nan error')
    #     exit()
    return c_64.float()

def get_L1_dist(a,b):
    # print( 'a = ',a.size() )
    # print( 'b = ',b.size() )
    # tmp_a = a.repeat(1,b.size(0)).view(a.size(0),b.size(0),-1)
    tmp_a = a.repeat(1,b.size(0))
    
    c = torch.sum( torch.abs(tmp_a-b.view(1,-1)).view(a.size(0),b.size(0),-1) , dim = 2 )

    # std_c = torch.zeros_like( c )
    # for i in range( a.size(0) ):
    #     for j in range( b.size(0) ):
    #         std_c [i][j] = torch.sum( torch.abs( a[i] - b[j] ) )
    # assert(  torch.sum( torch.abs( std_c - c ) ) < 1e-5 )
    # return std_c
    return c

def get_cos_dist(a,b):
    norm_1 = torch.norm(a,dim=1).repeat(b.size(0),1).t()

    norm_2 = torch.norm(b,dim=1).repeat(a.size(0),1)
    
    tmp = torch.mm(a,b.t())
    c = torch.mm(a,b.t())/norm_1/norm_2
    return c

def get_MSE_dist(a,b):
    return  torch.sum( (a-b)*(a-b) , dim = 1)/a.size(1)