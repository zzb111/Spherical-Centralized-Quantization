from functools import total_ordering
from math import isnan, nan

from torch.functional import norm
from tool import out_float_tensor

from scipy.ndimage.measurements import label, mean
from func_dist import get_L2_dist, get_cos_dist
from PIL.Image import PERSPECTIVE
import torch
import time

from torch._C import is_grad_enabled
from torch.nn.functional import embedding

# from torch._C import TreeView

def get_embedd(data_loader,net,model_setting):
    Is_single_label = model_setting['Is_single_label']
    n_data = data_loader.dataset.__len__()
    
    n_class = model_setting['n_class']
    device = model_setting['device']
    len_code = model_setting['len_code']

    with torch.no_grad():
        data_database = torch.zeros(n_data,len_code).to(device)
        label_database = torch.LongTensor(n_data).to(device)
        st = 0

        if ( Is_single_label == False):
            base = torch.ones( n_class , 1 ).to(device)
            for i in range(n_class):
                base[i][0] = 1 << i

        for i, data in enumerate(data_loader, 0):
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)

            image_embeddings , _ = net(images)

            ed = st + image_embeddings.size(0)
            
            data_database[st:ed] = image_embeddings[0:]

            if ( Is_single_label ):
                label_database[st:ed] = labels[0:]
            else:
                state = torch.mm( labels , base ).view(-1).long()
                label_database[st:ed] = state[0:]
            st = ed
            if ( i %200 ==0 ):
                print('get embedd ',i)
        print('get embedd OK')
    return data_database,label_database

def test_PL(database_loader,list_net,model_setting):

    Is_single_label = model_setting['Is_single_label']
    label_embeddings = model_setting['label_embeddings']
    device = model_setting['device']
    n_database_set = model_setting['n_database_set']
    
    deep_quan = len( list_net ) - 1
    n_class = label_embeddings.size(0)
    len_code = label_embeddings.size(1)
    len_subcode = len_code // deep_quan
    
    rebuild_database = torch.zeros( n_database_set , deep_quan , len_subcode ).to(device)
    
    label_database = torch.IntTensor(n_database_set).to(device)

    id_st = 0

    with torch.no_grad():

        base = torch.ones( n_class , 1 ).to(device)
        if ( not Is_single_label ):
            for i in range(n_class):
                base[i][0] = 1 << i

        for i, data in enumerate(database_loader, 0):
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)

            if ( Is_single_label == False ):
                state = torch.mm( labels , base ).view(-1).long()

            image_embeddings , _ = list_net[0](images)

            id_end = id_st + image_embeddings.size(0)

            if ( Is_single_label ):
                label_database[id_st:id_end] = labels[0:]
            else:
                label_database[id_st:id_end] = state[0:]

            aim_hard = torch.zeros( deep_quan , image_embeddings.size(0) , len_subcode ).to(device)

            for j in range(deep_quan):
                aim_hard[j] += image_embeddings[ : , j*len_subcode : (j+1)*len_subcode ]

            for j in range(1,len(list_net)):
                Q_hard, _ = list_net[j]( aim_hard[j-1] )
                rebuild_database[ id_st : id_end , j-1] += Q_hard[:,:]

            id_st = id_end

            if ( i % 200 ==0 ):
                print('database build ',i)
    rebuild_database = rebuild_database.view(n_database_set ,-1)
    return rebuild_database, label_database



def test_mAP_org(list_dataloader,net, model_setting):
    print('test org')
    Is_single_label = model_setting['Is_single_label']

    n_query_set = model_setting['n_query_set']
    n_database_set = model_setting['n_database_set']
    n_rank = model_setting['n_rank']
    
    n_class = model_setting['n_class']
    device = model_setting['device']
    batch_query = model_setting['batch_query']
    len_code = model_setting['len_code']

    query_loader = list_dataloader['query_loader']
    database_loader = list_dataloader['database_loader']

    total = 0
    query_iter = (n_query_set//batch_query)//20

    pre = time.time()

    with torch.no_grad():
        mean_acc = 0.0
        
        data_database = torch.zeros(n_database_set,len_code).to(device)

        label_database = torch.IntTensor(n_database_set).to(device)
        pos = torch.FloatTensor( range(n_rank) ).to( device ) + 1

        zeroooo = torch.zeros( n_database_set ).to(device)
        oneeee = torch.ones( n_database_set ).to(device)
        st = 0

        if ( Is_single_label == False):
            base = torch.ones( n_class , 1 ).to(device)
            for i in range(n_class):
                base[i][0] = 1 << i

        for i, data in enumerate(database_loader, 0):
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)

            image_embeddings , _ = net(images)

            ed = st + image_embeddings.size(0)
            
            data_database[st:ed] = image_embeddings[0:]

            if ( Is_single_label ):
                label_database[st:ed] = labels[0:]
            else:
                state = torch.mm( labels , base ).view(-1).long()
                label_database[st:ed] = state[0:]


            st = ed
            if ( i %200 ==0 ):
                print('build ',i)

        label_database = label_database.repeat( batch_query , 1 )
        norm_database = torch.norm( data_database , p=2,dim=1)
        norm_database , _ = torch.sort(norm_database,dim = 0 )
        id = 0 
        # eps = 0.2
        # st = 0
        # while ( st<15):
        #     last = id
        #     while ( ( id<norm_database.size(0) ) and ( norm_database[id] <st ) ):
        #         id+=1
        #     print('[ %.1f , %.1f) = %d '%(st,st+eps,id-last) )
        #     st += eps


        print('Database Build OK')

        for i, data in enumerate(query_loader, 0):
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            
            image_embeddings , _ = net(images)

            if ( Is_single_label == False):
                bit_labels = torch.mm( labels , base ).int()

            now_dist = torch.zeros(batch_query, n_database_set).to(device)

            now_dist = - torch.mm( image_embeddings , data_database.t() )
            # now_dist = - torch.mm( image_embeddings.double() , data_database.double().t() )

            new_id = torch.argsort( now_dist.view(batch_query,-1) , dim = 1 )
            now_label = torch.gather( label_database , 1 , new_id  )
            if (Is_single_label):
                cnt_take = torch.where( now_label == labels.view(-1,1) , oneeee , zeroooo  )
            else:
                cnt_take = torch.where(  ( ( now_label & bit_labels.view(-1,1) ) != 0 ) , oneeee , zeroooo  )

            # if ( i==0 ):
            #     aim_id = 15920
            #     print( now_dist[10][aim_id].item() )
            #     print('query label = ',label[10])
            #     print( label_database[10][aim_id].item() )
            #     print( torch.norm( data_database[aim_id], p=2,dim=0) )
                # print( new_id[10][3].item() )
                # n_dist = torch.gather( now_dist , 1 , new_id )
                # print( n_dist[10][3].item() )
                # print( cnt_take[10][3].item() )

            cnt_take = cnt_take[ : , 0:n_rank ]
            sum_take = torch.cumsum( cnt_take , 1 )
            all_take = cnt_take.sum(dim=1)
            all_take = torch.clamp( all_take , min = 1e-7  )
            now_acc = torch.sum( 
            torch.div(  sum_take , pos )
            *cnt_take , dim=1 
            )/all_take

            now_acc = now_acc.sum().item()
            mean_acc = mean_acc + now_acc
            total += batch_query

            if ( i%query_iter == query_iter-1 ):
                print('query finished %d  mean_acc = %.5f'%(total,mean_acc/total))

    mean_acc /= total
    print('org All mean_acc = %.5f'%(mean_acc))
    time_elapsed = time.time() - pre
    print('Test complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    return mean_acc

    
    




def test_mAP(list_dataloader,list_net,model_setting):
    print('test all')
    Is_single_label = model_setting['Is_single_label']

    n_query_set = model_setting['n_query_set']
    n_database_set = model_setting['n_database_set']
    n_rank = model_setting['n_rank']
    
    n_class = model_setting['n_class']
    device = model_setting['device']
    batch_query = model_setting['batch_query']
    len_code = model_setting['len_code']
    
    list_net[0].eval()

    query_loader = list_dataloader['query_loader']
    st_acc_list = 0

    pre = time.time()

    with torch.no_grad():
        rebuild_database , label_database = test_PL(list_dataloader['database_loader'],list_net, model_setting)
        if ( Is_single_label == False ):
            base = torch.ones( n_class , 1 ).to(device)
            for i in range(n_class):
                base[i][0] = 1 << i

        pos = torch.FloatTensor( range(n_rank) ).to( device ) + 1
        zeroooo = torch.zeros( n_database_set ).to(device)
        oneeee = torch.ones( n_database_set ).to(device)
        
        # rebuild_database = rebuild_database.view( n_database_set , len_code )

        print('database build OK')
        label_database = label_database.repeat( batch_query , 1 )
        
        query_iter = (n_query_set//batch_query)//20

        mean_acc = 0.0
        total = 0    
    

        for i, data in enumerate(query_loader, 0):
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            image_embeddings , _ = list_net[0](images)

            if ( Is_single_label == False ):
                bit_labels = torch.mm( labels , base ).int()

            now_dist = - torch.mm( image_embeddings , rebuild_database.t() )
            

            new_id = torch.argsort( now_dist.view(batch_query,-1) , dim = 1 )
            now_label = torch.gather( label_database , 1 , new_id  )

            if ( Is_single_label ):
                cnt_take = torch.where( now_label == labels.view(-1,1) , oneeee , zeroooo  )
            else:
                cnt_take = torch.where(  ( (now_label & bit_labels.view(-1,1) ) != 0 ) , oneeee , zeroooo  )

            cnt_take = cnt_take[ : , 0:n_rank ]
            sum_take = torch.cumsum( cnt_take , 1 )
            all_take = cnt_take.sum(dim=1)
            all_take = torch.clamp( all_take , min = 1e-17  )
            now_acc = torch.sum( 
            torch.div(  sum_take , pos )
            *cnt_take , dim=1
            )/all_take

            st_acc_list += now_acc.size(0)

            # for j in range( images.size(0) ):
            #     if ( now_acc[j] == 1.0 ):
            #         for k in range(10):
            #             print(new_id[j][k].item(), end=' ')
            #         print('')
            #     else:
            #         print('==================')
            # exit()

            now_acc = now_acc.sum().item()
            mean_acc = mean_acc + now_acc
            total += images.size(0)


            if ( i%query_iter == query_iter-1 ):
                print('query finished %d  mean_acc = %.5f'%(total,mean_acc/total))

    mean_acc = mean_acc / total
    print('new Accuracy of the network on the 10000 test images: %.3f%%' % (100 *mean_acc))
    time_elapsed = time.time() - pre
    print('Test complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    return mean_acc

def test_code_dis(list_dataloader,list_net,model_setting):

    Is_single_label = model_setting['Is_single_label']
    label_embeddings = model_setting['label_embeddings']
    device = model_setting['device']

    n_codeword = model_setting['n_codeword']


    database_loader = list_dataloader['database_loader']
    # train_loader = list_dataloader['train_loader']
    
    # is_priv = True
    # is_priv = False
    # print('is_priv = ',is_priv)
    
    deep_quan = len( list_net ) - 1
    n_class = label_embeddings.size(0)
    len_code = label_embeddings.size(1)
    len_subcode = len_code // deep_quan
    
    cnt_lable_code = torch.zeros( deep_quan, n_codeword , n_class ).to(device)
    cnt_pic_code = torch.zeros( deep_quan, n_codeword ).to(device)


    for deep in range(deep_quan):
        cnt_diff_code = 0
        for j in range(n_codeword):
            flag = False
            for k in range(j):
                if ( torch.norm(list_net[deep+1].CodeBook[j] - list_net[deep+1].CodeBook[k],p=2,dim=0) < 0.1):
                    flag = True
                    break
            if ( not flag ):
                cnt_diff_code += 1
        print('cnt_diff_code = ',cnt_diff_code)

    with torch.no_grad():

        base = torch.ones( n_class , 1 ).to(device)
        if ( not Is_single_label ):
            for i in range(n_class):
                base[i][0] = 1 << i

        sum_mse = 0.0
        cnt_data = 0.0

        for i, data in enumerate(database_loader, 0):
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)

            cnt_data += images.size(0)

            image_embeddings , _ = list_net[0](images)

            aim_hard = torch.zeros( deep_quan , image_embeddings.size(0) , len_subcode ).to(device)

            for j in range(deep_quan):
                aim_hard[j] += image_embeddings[ : , j*len_subcode : (j+1)*len_subcode ]

            for j in range(1,len(list_net)):
                quan_vec , code = list_net[j]( aim_hard[j-1] )
                # sum_mse += torch.norm( quan_vec - aim_hard[j-1] , p=2, dim=1 ).sum()
                tmp = ( quan_vec - aim_hard[j-1])
                sum_mse += torch.sum(tmp*tmp)
                # print( code )
                # print( labels )
                if ( Is_single_label ):
                    for k in range( code.size(0) ):
                        cnt_lable_code[ j-1 ][ code[k] ][ labels[k] ] += 1
                        cnt_pic_code[j-1][ code[k] ] += 1
                else:
                    for k in range( code.size(0) ):
                        for l in range( labels.size(0) ):
                            if ( labels[k][l] == 1 ):
                                cnt_lable_code[ j-1 ][ code[k] ][ l ] += 1
                        cnt_pic_code[j-1][ code[k] ] += 1

            if ( i % 200 ==0 ):
                print('database build ',i)
    print('mse = ',(sum_mse/cnt_data).item())

            

    CB_out= open('codebook.txt','w')

    for deep in range( deep_quan ):
        print('deep = ',deep,end=' ')
        mean_select = 0.0
        for k in range( n_codeword ):
            mean_select += cnt_pic_code[deep][k].item()
        mean_select /= n_codeword

        var = 0.0
        for k in range( n_codeword ):
            tmp = (cnt_pic_code[deep][k].item()-mean_select)
            var += tmp*tmp
        var /= n_codeword
        print('mean = %.3f var=%.3f'%(mean_select,var))

    priv_list = model_setting['priv_list']
    for i in range(deep_quan):
        main_class = [0]*n_codeword
        for j in range(n_codeword):
            if ( cnt_pic_code[i][j] > 0 ):
                for k in range( n_class ):
                    if ( cnt_lable_code[i][j][k] > cnt_lable_code[i][j][ main_class[j] ] ):
                        main_class[j] = k
        list_main_code = []
        for j in range( n_codeword ):
            if ( cnt_pic_code[i][j] > 0 ):
                list_main_code.append( (main_class[j],j) )
        list_main_code.sort()
        print('deep = ',i)

        for n_rank in [1,2,5,10,20,50,100,256]:
            mAp = 0.0
            Precision = 0.0
            with torch.no_grad():
                query_loader = list_dataloader['query_loader']
                cnt_query = 0
                for _, data in enumerate(query_loader, 0):
                    images, labels = data
                    images = images.to(device)
                    labels = labels.to(device)

                    cnt_query += images.size(0)

                    image_embeddings , _ = list_net[0](images)

                    image_embeddings = image_embeddings[:,i*len_subcode:(i+1)*len_subcode]

                    batch_query = image_embeddings.size(0)

                    now_dist = -torch.mm( image_embeddings , list_net[i+1].CodeBook.t() )
                    new_id = torch.argsort( now_dist.view(batch_query,-1) , dim = 1 )
                    for j in range(batch_query):
                        Ap = 0.0
                        cnt_ac = 0
                        for k in range(n_rank):
                            id = new_id[j][k]
                            if ( main_class[id] == labels[j] ):
                                cnt_ac += 1
                                Ap += cnt_ac/(k+1)
                                Precision += 1/n_rank

                        if ( cnt_ac >0 ):
                            mAp += Ap/cnt_ac
            mAp /= cnt_query
            Precision /= cnt_query
            print('mAp@%d = %.3f'%(n_rank,mAp),end=' ')
            print('Precision@%d = %.3f'%(n_rank,Precision),end=' ')
        print('')

        
        # if (is_priv):
        #     for k in range(n_class):
        #         st, ed = priv_list[k]
        #         for j in range(st,ed):
        #             cnt_correct += cnt_lable_code[i][j][k]
        # else:
        #     for j in range(n_codeword):
        #         cnt_correct += cnt_lable_code[i][j][ main_class[j] ]
        print('correct code_vec = ',end=' ')
        for Threshold in [0.8,0.9,0.95,0.99,0.995,0.999]:
            cnt_correct = 0
            for j in range(n_codeword):
                if ( cnt_lable_code[i][j][ main_class[j] ] > cnt_pic_code[i][j]*Threshold ):
                    cnt_correct += cnt_lable_code[i][j][ main_class[j] ].item()
            print('cnt@ %.3f = %d'%(Threshold,cnt_correct),end=' ')
        print('')
        n_Threshold = 10
        mean_right = 0.0
        for j in range(n_codeword):
            if ( cnt_pic_code[i][j] > 0 ):
                cnt_main = cnt_lable_code[i][j][ main_class[j] ].item()
                rate =  cnt_main / cnt_pic_code[i][j]
                mean_right += cnt_main * rate
        print('mean_right = %.3f'%(mean_right))
        


        for id in range(n_Threshold):
            st = id/n_Threshold
            ed = (id+1)/n_Threshold
            cnt = 0
            for j in range(n_codeword):
                # if ( ( cnt_lable_code[i][j][ main_class[j] ] > cnt_pic_code[i][j]*st ) and 
                    # ( cnt_lable_code[i][j][ main_class[j] ] <= cnt_pic_code[i][j]*ed ) ):
                if ( cnt_lable_code[i][j][ main_class[j] ] > cnt_pic_code[i][j]*st ):
                    cnt +=  cnt_lable_code[i][j][ main_class[j] ].item()
            # print('cnt_codeword in ( %.3f , %.3f ] = %d'%(st,ed,cnt))
            print('cnt@ %.3f = %d'%(st,cnt))

        
        # cnt_empty = 0
        # for j in range(n_codeword):
        #     if ( cnt_pic_code[i][j] <0  ):
        #         cnt_correct += cnt_lable_code[i][j][ main_class[j] ].item()
        # print('end=' ', = ',cnt_empty)
        # continue

        # label_embeddings = model_setting['label_embeddings'].view(n_class,deep_quan,len_subcode)

        # for c_id,ID in list_main_code:
        #     print('ID = %d class = %d'%(ID,c_id),end=' ')
        #     print('sum = ',cnt_pic_code[i][ ID ].item(),end=' ')
        #     print('dist = %.3f '%(torch.norm(label_embeddings[ c_id ][i][:] - list_net[i+1].CodeBook[ID][:] , p=2,dim=0).item()),end='')
        #     cnt_lable_code[i][ ID ] /= cnt_pic_code[i][ ID ]
        #     for k in range( n_class ):
        #         if ( cnt_lable_code[i][ ID ][k] > 0.01 ):
        #             print('tag = %d rate = %.3f '%( k , cnt_lable_code[i][ ID ][k] ),end=' ')
        #     print('')
        
        # for iiii in range( len(list_main_code) ):
        #     c_id_i, ID_i = list_main_code[iiii]
        #     codevec_i = list_net[i+1].CodeBook[ID_i]
        #     print('ID = %d class = %d '%( ID_i ,c_id_i ),end=' ')
        #     for j in range(iiii):
        #         c_id_j , ID_j = list_main_code[j]
        #         codevec_j = list_net[i+1].CodeBook[ID_j]
        #         print('dist to %d (c= %d) = %.3f '%(ID_j,c_id_j, torch.norm( codevec_i - codevec_j,p=2,dim=0 ).item() ),end=' ')
        #     print('')
            # if ( iiii > 20 ):
                # break
        
        # CB_out.write('deep = '+str(i)+'\n')
        # for c_id,id in list_main_code:
        #     CB_out.write('ID = '+str(id)+' C_ID = '+str(c_id)+'\n')
        #     for j in range(len_subcode):
        #         CB_out.write('%.3f '%(list_net[ i+1 ].CodeBook[id][j]))
        #     CB_out.write('\n')
        # CB_out.write('\n')

    # CB_out.close()
    return cnt_pic_code

def test_mse(data_load,list_net,model_setting):
    list_net[0].eval()

    Is_single_label = model_setting['Is_single_label']
    label_embeddings = model_setting['label_embeddings']
    device = model_setting['device']

    deep_quan = len( list_net ) - 1
    n_class = label_embeddings.size(0)
    len_code = label_embeddings.size(1)
    len_subcode = len_code // deep_quan

    with torch.no_grad():

        base = torch.ones( n_class , 1 ).to(device)
        if ( not Is_single_label ):
            for i in range(n_class):
                base[i][0] = 1 << i

        sum_mse = 0.0
        cnt_data = 0.0

        for i, data in enumerate(data_load, 0):
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)

            cnt_data += images.size(0)

            image_embeddings , _ = list_net[0](images)

            aim_hard = torch.zeros( deep_quan , image_embeddings.size(0) , len_subcode ).to(device)

            for j in range(deep_quan):
                aim_hard[j] += image_embeddings[ : , j*len_subcode : (j+1)*len_subcode ]

            for j in range(1,len(list_net)):
                quan_vec , _ = list_net[j]( aim_hard[j-1] )
                tmp = ( quan_vec - aim_hard[j-1])
                sum_mse += torch.sum(tmp*tmp)

            if ( i % 200 ==0 ):
                print('database build ',i)
    print('mse = ',(sum_mse/cnt_data).item())
    return (sum_mse/cnt_data).item()


def test_knn(list_dataloader,list_net,model_setting):
    list_net[0].eval()

    Is_single_label = model_setting['Is_single_label']
    label_embeddings = model_setting['label_embeddings']
    device = model_setting['device']
    batch_query = model_setting['batch_query']


    database_loader = list_dataloader['database_loader']
    query_loader = list_dataloader['query_loader']

    deep_quan = len( list_net ) - 1
    n_class = label_embeddings.size(0)
    len_code = label_embeddings.size(1)

    n_database = database_loader.dataset.__len__()

    embedding_db = torch.zeros(n_database,len_code).to(device)
    label_db = torch.IntTensor(n_database).to(device)

    total = 0
    N_knn = 5000
    mean_recall = 0.0
    with torch.no_grad():
        zeroooo = torch.zeros( n_database ).to(device)
        oneeee = torch.ones( n_database ).to(device)
        base = torch.ones( n_class , 1 ).to(device)
        if ( not Is_single_label ):
            for i in range(n_class):
                base[i][0] = 1 << i

        embedding_db, label_db = get_embedd(database_loader,list_net[0],model_setting)
        label_db = label_db.repeat( batch_query , 1 )
        for i, data in enumerate(query_loader, 0):
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)

            image_embeddings , _ = list_net[0](images)

            if ( Is_single_label == False):
                bit_labels = torch.mm( labels , base ).int()

            # cos_dist = get_cos_dist( image_embeddings , embedding_db)
            # dist = get_L2_dist( image_embeddings , embedding_db)
            dist = -torch.mm( image_embeddings , embedding_db.t())
            
            new_id = torch.argsort( dist.view( images.size(0) ,-1) , dim = 1 )
            now_label = torch.gather( label_db , 1 , new_id  )

            if (Is_single_label):
                cnt_take = torch.where( now_label == labels.view(-1,1) , oneeee , zeroooo  )
            else:
                cnt_take = torch.where(  ( ( now_label & bit_labels.view(-1,1) ) != 0 ) , oneeee , zeroooo  )

            recall_list = cnt_take[:,:N_knn]
            mean_recall += recall_list.sum()
            total += N_knn*labels.size(0)
    mean_recall /= total
    print('Recall@%d = %.5f'%(N_knn,mean_recall))
    return mean_recall


def test_gather(list_dataloader,list_net,model_setting):
    list_net[0].eval()

    Is_single_label = model_setting['Is_single_label']
    label_embeddings = model_setting['label_embeddings']
    device = model_setting['device']

    batch_query = model_setting['batch_query']
    n_database_set = model_setting['n_database_set']


    database_loader = list_dataloader['database_loader']
    query_loader = list_dataloader['query_loader']
    
    n_class = label_embeddings.size(0)
    len_code = label_embeddings.size(1)

    n_database = database_loader.dataset.__len__()

    embedding_db = torch.zeros(n_database,len_code).to(device)
    label_db = torch.IntTensor(n_database).to(device)
    

    id_st = 0
    with torch.no_grad():
        zeroooo = torch.zeros( n_database_set ).to(device)
        oneeee = torch.ones( n_database_set ).to(device)
        base = torch.ones( n_class , 1 ).to(device)

        if ( not Is_single_label ):
            for i in range(n_class):
                base[i][0] = 1 << i

        embedding_db, label_db = get_embedd(database_loader,list_net[0],model_setting)
        
        if ( Is_single_label ):
            tmp_label_db = ( label_db ).view(-1,1)
            cos_core_db = get_cos_dist( embedding_db , label_embeddings)
            aim_cos_core_db = torch.gather( cos_core_db , 1 , tmp_label_db )
            print('mean cos to core = ',aim_cos_core_db.mean().item())
        
        norm = torch.norm(embedding_db,p=2,dim=1)
        mean_norm = norm.mean()
        std_norm = torch.mean( (norm-mean_norm) ** 2 )
        print('mean_norm = ',mean_norm.item())
        print('std_norm = ',std_norm.item())

        sim_arr = []
        un_sim_arr = []
        lunkuo_index = 0.0
        total = 0
        for i, data in enumerate(query_loader, 0):
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)

            image_embeddings , _ = list_net[0](images)

            if ( Is_single_label == False):
                bit_labels = torch.mm( labels , base ).int()

            # dist = get_cos_dist( image_embeddings , embedding_db)
            dist = torch.sqrt( get_L2_dist( image_embeddings , embedding_db) )
            # dist = -torch.mm( image_embeddings.double() , embedding_db.double().t()).float()

            if (Is_single_label):
                cnt_take = torch.where( label_db == labels.view(-1,1) , oneeee , zeroooo  )
            else:
                cnt_take = torch.where(  ( ( label_db & bit_labels.view(-1,1) ) != 0 ) , oneeee , zeroooo  )
            
            sim_id = torch.argsort( cnt_take , dim = 1 )
            now_dist = torch.gather( dist , 1 , sim_id  )

            sum = torch.sum(cnt_take, dim = 1)
            for j in range(batch_query):
                cnt_sim = int(sum[j])
                un_sim_arr += (now_dist[j][:-cnt_sim]).tolist()
                sim_arr += (now_dist[j][-cnt_sim:]).tolist()
                
                mean_dis =  (now_dist[j][:-cnt_sim]).mean()
                mean_sim = (now_dist[j][-cnt_sim:]).mean()
                lunkuo_index +=  ( mean_dis - mean_sim)/max( mean_sim , mean_dis )
            total += labels.size(0)
        lunkuo_index /= total
        print('total = ',total)
        print('lunkuo_index = ',lunkuo_index.item())
    return sim_arr, un_sim_arr , mean_norm


def get_distinct_codewords(list_net):
    cnt = 0
    for deep in range(1,len(list_net)):
        for i in range( list_net[deep].CodeBook.size(0) ):
            tag = 1
            for j in range( list_net[deep].CodeBook.size(0) ):
                if ( i!=j ):
                    if (  torch.norm( list_net[deep].CodeBook[i]-list_net[deep].CodeBook[j] , p=2, dim =0 )<1e-2):
                        tag = 0
                        break
            cnt += tag
    return cnt

def get_cos_ap(list_dataloader,list_net,model_setting,is_org,aim_query):
    pre = time.time()
    list_net[0].eval()
    net = list_net[0]

    Is_single_label = model_setting['Is_single_label']
    label_embeddings = model_setting['label_embeddings']
    device = model_setting['device']
    n_rank = model_setting['n_rank']


    database_loader = list_dataloader['database_loader']
    # query_loader = list_dataloader['query_loader']
    new_batch_query = 200
    new_query_loader = torch.utils.data.DataLoader(database_loader.dataset , new_batch_query , shuffle=True, num_workers = 4 )
    # print(' num - new_query_loader = ',new_query_loader.dataset.__len__() )


    deep_quan = len( list_net ) - 1
    n_class = label_embeddings.size(0)
    len_code = label_embeddings.size(1)


    n_database = database_loader.dataset.__len__()

    query_iter = (aim_query//new_batch_query)//20

    embedding_db = torch.zeros(n_database,len_code).to(device)
    label_db = torch.IntTensor(n_database).to(device)

    total = 0
    mean_acc = 0.0
    list_ap = []
    with torch.no_grad():
        zeroooo = torch.zeros( n_database ).to(device)
        oneeee = torch.ones( n_database ).to(device)
        base = torch.ones( n_class , 1 ).to(device)
        pos = torch.FloatTensor( range(n_rank) ).to( device ) + 1

        if ( not Is_single_label ):
            for i in range(n_class):
                base[i][0] = 1 << i
        
        if is_org :
            embedding_db, label_db = get_embedd(database_loader,list_net[0],model_setting)
        else:
            embedding_db, label_db = test_PL(database_loader,list_net,model_setting)
        # print(embedding_db.size())
        # print(label_db.size())

        cluster_core = torch.zeros( n_class , len_code ).to(device)
        cnt_point = torch.zeros( n_class ).to(device)
        for i in range( embedding_db.size(0) ):
            cluster_core[ label_db[i] ] += embedding_db[i]
            cnt_point[ label_db[i] ] += 1
        cluster_core /= cnt_point.view(-1,1)

        label_db = label_db.repeat( new_batch_query , 1 )


        for i, data in enumerate(new_query_loader, 0):
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            
            image_embeddings , _ = net(images)

            if ( Is_single_label == False):
                bit_labels = torch.mm( labels , base ).int()

            now_dist = torch.zeros(new_batch_query, n_database).to(device)

            now_dist = - torch.mm( image_embeddings , embedding_db.t() )

            new_id = torch.argsort( now_dist.view(new_batch_query,-1) , dim = 1 )
            now_label = torch.gather( label_db , 1 , new_id  )
            if (Is_single_label):
                cnt_take = torch.where( now_label == labels.view(-1,1) , oneeee , zeroooo  )
            else:
                cnt_take = torch.where(  ( ( now_label & bit_labels.view(-1,1) ) != 0 ) , oneeee , zeroooo  )


            cnt_take = cnt_take[ : , 0:n_rank ]
            sum_take = torch.cumsum( cnt_take , 1 )
            all_take = cnt_take.sum(dim=1)
            all_take = torch.clamp( all_take , min = 1e-7  )
            now_acc = torch.sum( 
            torch.div(  sum_take , pos )
            *cnt_take , dim=1 
            )/all_take

            for j in range(now_acc.size(0)):
                cos = torch.cosine_similarity(image_embeddings[j],cluster_core[ labels[j] ],dim = 0)
                ap = now_acc[j]
                list_ap.append( ( cos , ap  ) )

            now_acc = now_acc.sum().item()
            mean_acc = mean_acc + now_acc
            total += new_batch_query

            if ( i%query_iter == query_iter-1 ):
                print('query finished %d  mean_acc = %.5f'%(total,mean_acc/total))
            if ( total >= aim_query ):
                print('break')
                break

    mean_acc /= total
    print('mean_acc = %.5f'%(mean_acc))
    time_elapsed = time.time() - pre
    print('Test complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    
    gap = 0.01
    ap_gp = torch.zeros( int(2/gap) ).to(device)
    cnt_gp  = torch.zeros( int(2/gap) ).to(device)

    for i in range( len(list_ap) ):
        cos , ap = list_ap[i]
        id = int( (cos+1) / gap)
        ap_gp [ id ] += ap
        cnt_gp[ id ] += 1
    for i in range( ap_gp.size(0) ):
        if ( cnt_gp[i] >0 ):
            ap_gp[i] /= cnt_gp[i]

    out_float_tensor(ap_gp)
    out_float_tensor(cnt_gp)

    return mean_acc, list_ap