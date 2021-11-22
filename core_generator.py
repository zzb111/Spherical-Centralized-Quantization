from numpy.core.fromnumeric import argmax
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from sys import argv

from torch.utils import data
from tool import out_float_tensor,load_float_tensor
from func_dist import get_L2_dist , get_cos_dist

def get_sim_matrix(dir_train):

    n_class = 21
    count_matrix = torch.zeros( n_class , n_class )
    count_class = torch.zeros(n_class)
    with open(dir_train, 'r') as f:
        for lines in f:
            tmp = lines.split()
            label_list = []
            for i in range(n_class):
                if ( tmp[i+1] == '1' ):
                    label_list.append(i)
                    count_class[i] += 1
            
            for i in range( len(label_list) ):
                for j in range( len(label_list) ):
                    count_matrix[ label_list[j] ] [  label_list[ i ] ] += 1
        f.close()

    for i in range(n_class):
        for j in range(n_class):
            count_matrix[i][j] = count_matrix[i][j] / ( count_class[i] + count_class[j] -count_matrix[i][j]  )
    return count_matrix



def generate_core(n_core,dim,sim_matrix,dir_core):
    n_epoch = 10000

    # margin = dim * (1- sim_matrix)
    margin = sim_matrix*2-1
    out_float_tensor(margin)

    core = torch.nn.Parameter ( torch.normal( mean = torch.zeros( n_core , dim ) , std = 2 ) )


    # print('core init')
    # out_float_tensor(core)

    optimizer = optim.SGD( [core] , lr=1, momentum=0.9 , nesterov=True )
    # optimizer = optim.adam( [core] , lr=1, momentum=0.9 , nesterov=True )
    # optimizer = optim.Adam( [core] , lr = 10 , amsgrad=True , weight_decay=1e-5 )
    scheduler = lr_scheduler.StepLR(optimizer,step_size=2000,gamma = 0.9 )

    mask = torch.BoolTensor( n_core , n_core )
    mask_ones = torch.ones( n_core , n_core)
    mask_zeros = torch.zeros( n_core , n_core)
    for i in range( n_core ):
        for j in range( n_core ):
            mask[i][j] = i!=j

    epoch_iter = n_epoch//10

    for epoch in range( n_epoch ):

        optimizer.zero_grad()

        # tmp_dist = get_L2_dist( core , core ) *mask
        # error = ( ( margin - tmp_dist )*( margin - tmp_dist ) ).sum() /(n_core*(n_core-1)/2)
        
        tmp_dist = get_cos_dist( core , core )
        masked_dist_zero = torch.where( mask , (margin-tmp_dist)*(margin-tmp_dist) , mask_zeros )

        error = torch.sum(  masked_dist_zero ) /(n_core*(n_core-1))
        max_error = torch.zeros(1)    
        for i in range( n_core ):
            max_id = torch.argmax( masked_dist_zero[i] , dim = 0 )
            max_error += masked_dist_zero[i][max_id]
            # max_error += tmp_dist[i][max_id]
        max_error /= n_core

        loss = error + max_error
        # loss = error

        loss.backward()

        optimizer.step()
        scheduler.step()
        if ( epoch % epoch_iter == epoch_iter -1 ):
            print('loss =  %.5f  error = %.5f ' % ( loss.item() , error.item() ) )
    core_norm = torch.norm( core , dim=1 , p =2 )
    print('before Normalization  max = %.3f min=%.3f'%(core_norm.max() , core_norm.min() ))
    print('Finished.')
    core.data /= core_norm.view(-1,1)
    core_norm = torch.norm( core , dim=1 , p =2 )
    print('after Normalization  max = %.3f min=%.3f'%(core_norm.max() , core_norm.min() ))

    # print('L2 dist')
    # core_sim = get_L2_dist( core , core )
    # out_float_tensor(core_sim)

    # print('cos dist')
    core_sim = get_cos_dist( core , core )
    out_float_tensor(core_sim)
    print('sum cos dist')
    sum = torch.sum(core_sim , dim = 1)
    out_float_tensor(sum)

    out_float_tensor( margin - core_sim)

    # out_float_tensor(core)

    # exit()
    with open(dir_core, 'w') as f:
        for i in range(n_core):
            for j in range(dim):
                f.write( str ( core[i][j].item() )+' ')
            f.write('\n')
        f.close()
    return core


def get_core(argv):
    model_setting = {}
    argv = argv[1:]
    for item in argv:
        a , b = item.split('=')
        model_setting[a] = b

    dir_train = model_setting['dir_train']
    dir_core = model_setting['dir_core']
    dir_sim = model_setting['dir_sim']
    dataset = model_setting['dataset']
    dim = int( model_setting['dim'] )

    if ( dataset =='CIFAR10' ):
        n_label = 10
    elif ( dataset =='ImageNet' ):
        n_label = 100
    elif ( dataset =='NUS' ):
        n_label = 21
    elif ( dataset =='Test'):
        n_label = 3

    if ( dir_sim == 'none' ):
        if ( dataset=='NUS'):
            sim_matrix =  get_sim_matrix(dir_train)
        else:
            sim_matrix = torch.eye(n_label)
    else:
        sim_matrix = load_float_tensor(dir_sim)
        

    # out_float_tensor(sim_matrix)
    core = generate_core(n_label,dim,sim_matrix,dir_core)

if __name__ == '__main__':
    get_core(argv)