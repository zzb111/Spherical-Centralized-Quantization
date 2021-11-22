
import torch
from torch._C import Size

def get_rebuild_vec( image_embeddings , list_net , device ):
    deep_quan = len( list_net )-1
    len_subcode = list_net[1].CodeBook.size(1)

    aim_hard = torch.zeros( deep_quan , image_embeddings.size(0) , len_subcode ).to(device)
    rebuild_vec = torch.zeros( image_embeddings.size(0) , deep_quan , len_subcode ).to(device)

    for j in range(deep_quan):
        aim_hard[j] += image_embeddings[ : , j*len_subcode : (j+1)*len_subcode ]
    
    for j in range(1,len(list_net)):
        Q_hard  = list_net[j]( aim_hard[j-1] )

        rebuild_vec[  : , j-1] += Q_hard[:,:]
    rebuild_vec= rebuild_vec.view( ( image_embeddings.size(0) ,-1  ) )
    return rebuild_vec

def IsFloat(str):
    try:
        tmp = float(str)
        return True
    except:
        return False
    # s=str.split('.')
    # if len(s)>2:
    #     return False
    # return (s[0].isdigit() and s[1].isdigit() )

def load_setting(f_path):
    M = {}
    with open(f_path, 'r') as f:
        while True:
            line_tmp = f.readline()
            if not line_tmp:
                break
            line_tmp = line_tmp.split()
            name = line_tmp[0]
            val = line_tmp[-1]
            if ( val.isdigit() ):
                M[ name ] = int( val )
            elif ( IsFloat(val) ):
                M[ name ] = float( val )
            elif( val =='True' ):
                M[ name ] = True
            elif( val =='False' ):
                M[ name ] = False
            else:
                M[ name ] = val
        f.close()
    return M


def load_label_embeddings(n_labels,len_feature,label_embeddings_path):
    label_embeddings = torch.zeros(n_labels*len_feature )
    id = 0
    with open(label_embeddings_path, 'r') as f:
        while True:
            line_tmp = f.readline()
            if not line_tmp:
                break
            line_tmp = line_tmp.split()
            for x in line_tmp:
                label_embeddings[id] = float(x)
                id=id+1
        f.close()
    label_embeddings = label_embeddings.view(n_labels,len_feature)
    return label_embeddings

def load_list_file(list_dir):
    out_list = []
    with open(list_dir, 'r') as f:
        line_tmp = f.readline()
        line_tmp = line_tmp.split()
        for x in line_tmp:
            out_list.append( int(x) )
        f.close()
    return out_list

def new_load_list_file(dir_list):
    list_data = []

    with open(dir_list, 'r') as f:
        for line in f:
            line = line.split()
            list_data.append( ( int(line[0]) , line[1] , int(line[2]) ) )
        f.close()
    return list_data

def write_list_file(in_list,list_dir):
    with open(list_dir, 'w') as f:
        for x in in_list:
            f.write(str(x)+' ')
        f.close()

def new_write_list_file(list_id,dir_list):
    with open(dir_list, 'w') as f:
        
        for i in range( len(list_id) ):
            f.write(str(list_id[i][0])+' '+str(list_id[i][1])+' '+str(list_id[i][2])+'\n')
        
        f.close()

def out_float_tensor(A):
    if ( A.dim() == 1 ):
        for i in range( A.size(0) ):
            print('%.3f'%(A[i]),end=' ')
        print('')
    else:
        assert(A.dim()==2)
        for i in range( A.size(0) ):
            for j in range( A.size(1)):
                print('%.3f'%(A[i][j]),end=' ')
            print('')

def load_float_tensor(dir):
    f = open(dir,'r')
    out = []
    for line in f:
        line = line.split()
        for i in range( len( line ) ):
            line[i] = float(line[i])
        out.append(line)
    return torch.FloatTensor(out)