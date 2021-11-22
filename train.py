
import time
from numpy.lib.type_check import imag
import torch

from loss import CE_loss, core_loss , new_MSE_loss, adaptive_margin_loss
from loss import cls_quan_loss, MAX_MIN_regular_loss
from tool import get_rebuild_vec

def opt_org(epoch,data_loader,net,list_optimizer,list_scheduler, model_setting):
    print('start org optimization epoch  %d  lr = %.5f'%( epoch ,list_optimizer[0].param_groups[-1]['lr']) )

    n_train_set = model_setting['n_train_set']
    random_noise_mean = model_setting['random_noise_mean']
    random_noise_stddev = model_setting['random_noise_stddev']
    org_cls_rate = 0.1
    n_class = model_setting['n_class']
    device = model_setting['device']
    label_embeddings = model_setting['label_embeddings']
    
    batch_size = model_setting['batch_size']

    running_loss = 0.0
    running_loss_1 = 0.0
    running_loss_2 = 0.0

    running_core_like = 0.0

    mean_loss = 0.0
    total = 0
    pre = time.time()

    iter_num =  ( n_train_set//batch_size )//2

    mean_cos_class = [0]*n_class

    mean_fea_norm = 0.0
    
    train_loader = data_loader['train_loader']
    for i, data in enumerate(train_loader, 0):
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)
        
        images = images + torch.normal( mean = random_noise_mean*torch.zeros_like(images) , std = random_noise_stddev*torch.ones_like(images) ).to(device)

        for optimizer in list_optimizer:
            optimizer.zero_grad()
        
        if ( model_setting['Train_core'] ):
            model_setting['core_optimizer'].zero_grad()
            model_setting['fea_to_cls_optimizer'].zero_grad()

        image_embeddings , predicts = net(images)

        loss_1 ,mean_core_like = core_loss(  image_embeddings , labels , label_embeddings  ,  model_setting )
        loss_2 = org_cls_rate*CE_loss( predicts,labels ,  model_setting )

        if ( model_setting['Train_core'] ):
            fea_predicts = model_setting['fea_to_cls']( image_embeddings )
            train_core_mse = model_setting['train_core_mse']
            tmp = 0
            for n in range( image_embeddings.size(0) ):
                tmp += torch.sum( torch.norm( image_embeddings[n] - label_embeddings[ labels[n]] , p=2,dim=0) )
            loss = train_core_mse*tmp + CE_loss( fea_predicts,labels ,  model_setting ) + loss_2
        elif ( model_setting['Train_dvsq'] ):
            loss = adaptive_margin_loss(image_embeddings,labels,label_embeddings,model_setting)+loss_2
        else:
            loss = loss_1 + loss_2
        
        running_loss += loss.item()
        running_loss_1 += loss_1.item()
        running_loss_2 += loss_2.item()

        running_core_like += mean_core_like.item()

        mean_loss += loss.item()*(labels.size(0))
        total += labels.size(0)
        mean_fea_norm += torch.norm(image_embeddings,p=2,dim=1).mean().item()


        loss.backward()

        for optimizer in list_optimizer:
            optimizer.step()

        if ( model_setting['Train_core'] ):
            model_setting['core_optimizer'].step()
            model_setting['fea_to_cls_optimizer'].step()

        if ( i %iter_num == iter_num-1 ):
            print('epoch  %d finished %d \n loss =  %.7f core_loss =  %.7f  loss_classfier = %.7f'%
            (epoch , i, running_loss/iter_num , running_loss_1/iter_num , running_loss_2/iter_num) )
            print('mean_core_like = %.7f  mean_fea_norm = %.3f'%(running_core_like/iter_num , mean_fea_norm/iter_num ))
            
            running_loss = 0.0
            running_loss_1 = 0.0
            running_loss_2 = 0.0
            mean_fea_norm = 0.0
            
            running_core_like = 0.0
            time_elapsed = time.time() - pre
            print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
            pre = time.time()
    
    list_scheduler[0].step()
    if ( model_setting['Train_core'] ):
        model_setting['core_scheduler'].step()
        model_setting['fea_to_cls_scheduler'].step()

    return mean_loss/total

def opt_code_cls(epoch,data_loader,list_net,list_optimizer,list_scheduler,to_cls, model_setting):
    n_train_set = model_setting['n_train_set']
    random_noise_mean = model_setting['random_noise_mean']
    random_noise_stddev = model_setting['random_noise_stddev']
    n_class = model_setting['n_class']
    device = model_setting['device']
    label_embeddings = model_setting['label_embeddings']
    
    
    batch_size = model_setting['batch_size']
    
    quan_cls_rate = model_setting['quan_cls_rate']
    quan_cls_norm_rate = model_setting['quan_cls_norm_rate']

    print('start code optimization by cls epoch ',epoch)
    running_loss = 0.0
    mean_loss_1 = .0
    mean_loss_2 = .0
    mean_loss_3 = .0

    mean_loss = 0.0
    total = 0
    pre = time.time()

    train_loader = data_loader['train_loader']

    iter_num =  ( n_train_set//batch_size )//2


    
    for i, data in enumerate(train_loader, 0):
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)
        
        images = images + torch.normal( mean = random_noise_mean*torch.zeros_like(images) , std = random_noise_stddev*torch.ones_like(images) ).to(device)

        for optimizer in list_optimizer:
            optimizer.zero_grad()

        image_embeddings , predicts = list_net[0](images)


        rebuild_vec =  get_rebuild_vec( image_embeddings , list_net , device )
        loss_1 , _ , _ , _ , _ , _ , _ = new_MSE_loss( image_embeddings , labels ,list_net ,  model_setting )
        loss_2 = quan_cls_rate*cls_quan_loss( labels , rebuild_vec , to_cls, device )
        loss_3 = quan_cls_norm_rate*torch.norm( to_cls.weight ).sum()
        
        loss = loss_1 + loss_2 + loss_3
        
        running_loss += loss.item()

        mean_loss += loss.item()*(labels.size(0))
        mean_loss_1 += loss_1.item()*(labels.size(0))
        mean_loss_2 += loss_2.item()*(labels.size(0))
        mean_loss_3 += loss_3.item()*(labels.size(0))


        total += labels.size(0)

        loss.backward()

        for j in range( 1 , len(list_optimizer) ):
            list_optimizer[j].step()

        if ( i %iter_num == iter_num-1 ):
            print('epoch  %d finished %d \n loss =  %.7f '% (epoch , i, running_loss/iter_num ) )
            print('loss_1 = %.5f loss_2 = %.5f loss_3 = %.5f'% ( mean_loss_1/iter_num , mean_loss_2 /iter_num 
            , mean_loss_3 /iter_num ) )
            
            running_loss = 0.0
            mean_loss_1 = 0.0
            mean_loss_2 = 0.0
            mean_loss_3 = 0.0

            time_elapsed = time.time() - pre
            print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
            pre = time.time()

    for j in range( 1 , len(list_scheduler) ):
        list_scheduler[j].step()

    tmp_norm = 0.0
    deep_quan = model_setting['deep_quan']
    for i in range( deep_quan ):
        tmp_norm += torch.sum( torch.norm( list_net[i+1].CodeBook , p=2 , dim = 1 ) ).item()
    print('norm of Codebook = ', tmp_norm )
    return mean_loss/total



def opt_code(epoch,data_loader,list_net,list_optimizer,list_scheduler,model_setting ):
    print('start code optimization epoch  %d  lr = %.5f'%( epoch ,list_optimizer[1].param_groups[-1]['lr']) )
    n_train_set = model_setting['n_train_set']
    random_noise_mean = model_setting['random_noise_mean']
    random_noise_stddev = model_setting['random_noise_stddev']
    device = model_setting['device']
    deep_quan = model_setting['deep_quan']
    
    quan_reg_rate = model_setting['quan_reg_rate']

    batch_size = model_setting['batch_size']

    running_loss = 0.0
    
    running_mean_dist_1 = 0.0
    running_mean_dist_2 = 0.0
    running_mean_dist_3 = 0.0

    running_mean_loss_2 = 0.0

    running_mean_priv_loss = 0.0
    running_mean_public_loss = 0.0
    running_mean_push_loss = 0.0
    running_mean_select_priv_loss = 0.0

    mean_loss = 0.0
    total = 0
    pre = time.time()

    iter_num =  ( n_train_set//batch_size )//2

    train_loader = data_loader['train_loader']
    for i, data in enumerate(train_loader, 0):
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)
        
        images = images + torch.normal( mean = random_noise_mean*torch.zeros_like(images) , std = random_noise_stddev*torch.ones_like(images) ).to(device)

        for optimizer in list_optimizer:
            optimizer.zero_grad()

        image_embeddings , _ = list_net[0](images)


        loss_1 , mean_dist_1 , mean_dist_2 , mean_dist_3 ,mean_priv_loss , mean_public_loss , mean_push_loss, mean_select_priv_loss = new_MSE_loss( 
        image_embeddings , labels  ,list_net ,  model_setting )

        loss_2 = quan_reg_rate * MAX_MIN_regular_loss(list_net,model_setting)

        loss = loss_1 + loss_2

        
        running_loss += loss.item()

        running_mean_priv_loss += mean_priv_loss
        running_mean_public_loss += mean_public_loss
        running_mean_push_loss += mean_push_loss
        running_mean_select_priv_loss += mean_select_priv_loss

        running_mean_loss_2 += loss_2.item()
        
        running_mean_dist_1 += mean_dist_1.item()
        running_mean_dist_2 += mean_dist_2.item()
        running_mean_dist_3 += mean_dist_3.item()

        mean_loss += loss.item()*(labels.size(0))
        total += labels.size(0)

        loss.backward()

        for j in range( 1 , len(list_optimizer) ):
            list_optimizer[j].step()

        if ( i %iter_num == iter_num-1 ):
            print('epoch  %d finished %d \n loss =  %.7f '% (epoch , i, running_loss/iter_num ) )
            print('mean_t1 = %.5f mean_t2 = %.5f  mean_t3 = %.5f '% ( running_mean_dist_1/iter_num , running_mean_dist_2 /iter_num   
            , running_mean_dist_3 /iter_num ) )
            print('priv_loss = %.5f public_loss = %.5f  push_loss = %.5f '% ( running_mean_priv_loss/iter_num , running_mean_public_loss /iter_num   
            , running_mean_push_loss /iter_num ) )
            print('running_mean_loss_2 = %5f running_mean_select_priv_loss = %.5f'%(running_mean_loss_2 /iter_num,
            running_mean_select_priv_loss/iter_num ))

            running_loss = 0.0
            running_mean_loss_2 = 0.0
            
            running_mean_dist_1 = 0.0
            running_mean_dist_2 = 0.0
            running_mean_dist_3 = 0.0

            running_mean_priv_loss = 0.0
            running_mean_public_loss = 0.0
            running_mean_push_loss = 0.0
            running_mean_select_priv_loss = 0.0

            time_elapsed = time.time() - pre
            print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
            pre = time.time()

    for j in range( 1 , len(list_scheduler) ):
        list_scheduler[j].step()

    tmp_norm = 0.0
    for i in range( deep_quan ):
        tmp_norm += torch.sum( torch.norm( list_net[i+1].CodeBook , p=2 , dim = 1 ) ).item()
    print('norm of Codebook = ', tmp_norm )
    return mean_loss/total