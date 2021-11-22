
import time
from scipy.ndimage.measurements import label
from scipy.sparse.linalg import isolve
import torch
from torch.nn.functional import embedding
# from torch._C import FloatTensor
from torch.utils.data import dataset
import torchvision
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import transforms
from sys import argv

from model import model_feature, model_quantization , model_feature_resnet

from tool import load_label_embeddings, load_setting
from get_dataset import prepare_CIFAR10 , prepare_CIFAR10_datatrain , prepare_ImageNet
from dataset import my_NUS
from train import opt_org , opt_code
from test import test_mAP_org , test_mAP, test_code_dis
from test import test_mse,test_gather,test_knn,get_distinct_codewords,get_cos_ap
from tool import out_float_tensor

from loss import MAX_MIN_regular_loss

from func_dist import  get_cos_dist

def get_optimizers(list_net,model_setting):
    model_setting['random_noise_mean'] = 0.0
    model_setting['random_noise_stddev'] = 0.01
    base_lr = model_setting['learning_rate']
    decay_rate = 0.9
    decay_step = 10
    lr_code = model_setting['lr_code']
    lr_cls = 0.01
    fc_rate = 40
    fc_weight_decay = 1e-5

    Is_ResNet = (model_setting['backbone'] == 'ResNet')

    weight_decay = 0
    weight_decay_norm = 0
    weight_decay_bias = 0

    bias_lr_factor = 2.0

    norm_module_types = (
        torch.nn.BatchNorm1d,
        torch.nn.BatchNorm2d
    )

    cnt_fc = 0
    params: List[Dict[str, Any]] = []
    memo: Set[torch.nn.parameter.Parameter] = set()
    for module in list_net[0].modules():
        for module_param_name, value in module.named_parameters(recurse=False):
            if not value.requires_grad:
                continue
            # Avoid duplicating parameters
            if value in memo:
                print('memo continue ???')
                continue
            memo.add(value)

            schedule_params = {
                "lr": base_lr,
                "weight_decay": weight_decay,
            }
            if isinstance(module, norm_module_types):
                schedule_params["weight_decay"] = weight_decay_norm
            elif module_param_name == "bias":    
                schedule_params["lr"] = base_lr * bias_lr_factor
                schedule_params["weight_decay"] = weight_decay_bias

            if isinstance(module, (torch.nn.Linear) ):
                cnt_fc += 1
                if ( ( cnt_fc > 4 ) or ( Is_ResNet ) ):
                    if ( module_param_name == "bias" ):
                        schedule_params["lr"] = base_lr * bias_lr_factor * fc_rate
                    else:
                        schedule_params["lr"] = base_lr * fc_rate
                else:
                    if ( module_param_name == "bias" ):

                        schedule_params["lr"] = base_lr * bias_lr_factor
                    else:
                        schedule_params["lr"] = base_lr
                schedule_params["weight_decay"] = fc_weight_decay
                        
            params += [
                    {
                        "params": [value],
                        "lr": schedule_params["lr"],
                        "weight_decay": schedule_params["weight_decay"],
                    }
                ]

    list_optimizer = [] 
    list_scheduler = []

    optimizer = optim.SGD( params  , momentum=0.9 , nesterov= True)
    optimizer_fc = optim.SGD( params[-4:]  , momentum=0.9 , nesterov= True)

    scheduler = lr_scheduler.StepLR(optimizer,step_size=decay_step,gamma = decay_rate  )
    scheduler_fc = lr_scheduler.StepLR(optimizer_fc,step_size=decay_step,gamma = decay_rate  )

    list_optimizer.append(optimizer)
    list_scheduler.append(scheduler)


    if ( 'Train_core' in  model_setting ) and ( model_setting['Train_core'] ):
        lr_train_core = model_setting['lr_train_core']
        global tmp_label_embeddings
        core_optimizer = optim.SGD( [tmp_label_embeddings],lr= lr_train_core , momentum=0.9 , nesterov= True)
        core_scheduler = lr_scheduler.StepLR(optimizer,step_size=decay_step,gamma = decay_rate  )
        model_setting['core_optimizer'] = core_optimizer
        model_setting['core_scheduler'] = core_scheduler

        lr_fea_to_cls = model_setting['lr_fea_to_cls']
        global fea_to_cls
        fea_to_cls_optimizer = optim.SGD( fea_to_cls.parameters() ,lr= lr_fea_to_cls , momentum=0.9 , nesterov= True)
        fea_to_cls_scheduler = lr_scheduler.StepLR(optimizer,step_size=decay_step,gamma = decay_rate  )
        model_setting['fea_to_cls_optimizer'] = fea_to_cls_optimizer
        model_setting['fea_to_cls_scheduler'] = fea_to_cls_scheduler
    else:
        model_setting['Train_core'] = False


    quan_weight_decay = 1e-5
    for i in range(1,deep_quan+1):
        optimizer = optim.Adam( list_net[i].parameters() , lr = lr_code , amsgrad=True , weight_decay=quan_weight_decay )
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.99)

        list_optimizer.append(optimizer)
        list_scheduler.append(scheduler)


    optimizer = optim.SGD( [to_cls.weight,to_cls.bias] , momentum=0.9 , lr = lr_cls , nesterov= True)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=decay_step,gamma = decay_rate  )
    
    list_optimizer.append(optimizer)
    list_scheduler.append(scheduler)

    return list_optimizer,list_scheduler,optimizer_fc,scheduler_fc


def get_priv_list(model_setting):
    priv_rate = model_setting['priv_rate']
    n_class = model_setting['n_class']
    n_codeword = model_setting['n_codeword']

    quan_mode = 'Priv + public'
    priv_list = [0]*n_class

    if ( priv_rate*n_class > n_codeword ):
        quan_mode = 'Priv only'
        priv_rate = n_codeword // n_class +1
        cnt = 0
        cnt_id = 0
        for i in range( n_codeword % n_class ):
            priv_list[ cnt_id ] = ( cnt , cnt+priv_rate )
            cnt_id += 1
            cnt += priv_rate

        for i in range( n_codeword % n_class , n_class ):
            priv_list[ cnt_id ] = ( cnt , cnt+priv_rate-1 )
            cnt_id += 1
            cnt += priv_rate-1
    else:
        if ( priv_rate == 0 ):
            quan_mode = 'Public Public Public Public only'
        cnt = 0
        for i in range( n_class ):
            priv_list[i] = ( cnt , cnt+priv_rate )
            cnt += priv_rate
    
    model_setting['priv_rate'] = priv_rate
    model_setting['priv_list'] = priv_list
    model_setting['quan_mode'] = quan_mode
    print('priv_list =',priv_list)
    print('quan_mode =',quan_mode)

def get_dataset(model_setting):

    dataset_name = model_setting['dataset']
    test_setting = model_setting['test_setting']
    backbone = model_setting['backbone']

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])

    transform = transforms.Compose([
            torchvision.transforms.ColorJitter(brightness=0.3, contrast=0, saturation=0.1, hue=0),
            transforms.RandomResizedCrop( 224 ,scale=(0.7,1.3)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])


    val_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])

    if ( ( dataset_name == 'CIFAR10' ) and ( test_setting == '1') ):
        list_path = './dataset/CIFAR-10/setting-1/'
        org_train_set,train_set, query_set, database_set = prepare_CIFAR10(transform,val_transform,True,list_path)
        
        model_setting.update( load_setting('./settings/CIFAR10-1.txt') )
        
        label_embeddings = load_label_embeddings(10,360,"./dataset/CIFAR-10/setting-1/core.txt")
        label_embeddings = label_embeddings.to(device)
    elif ( ( dataset_name == 'CIFAR10' ) and ( test_setting == '2') ):
        list_path = './dataset/CIFAR-10/setting-2/'
        org_train_set,train_set, query_set, database_set = prepare_CIFAR10_datatrain(transform,val_transform,True,list_path)

        model_setting.update( load_setting('./settings/CIFAR10-2.txt') )
        
        label_embeddings = load_label_embeddings(10,360,"./dataset/CIFAR-10/setting-2/core.txt")
        label_embeddings = label_embeddings.to(device)
    elif ( ( dataset_name == 'NUS' ) and ( test_setting == '1') ):
        list_path = './dataset/nus-wide/setting-1/'
        train_set = my_NUS('train',  list_path , transform = transform )
        org_train_set = my_NUS('train', list_path , transform = val_transform )
        database_set = my_NUS('database', list_path, transform = val_transform)
        query_set = my_NUS('query', list_path, transform = val_transform)

        model_setting.update( load_setting('./settings/NUS-WIDE-1.txt') )

        label_embeddings_path = "./dataset/nus-wide/setting-1/core.txt"
        label_embeddings = load_label_embeddings(21,360,label_embeddings_path)
        model_setting['label_embeddings_path'] = label_embeddings_path
        label_embeddings = label_embeddings.to(device)

    elif ( ( dataset_name == 'NUS' ) and ( test_setting == '2') ):
        list_path = './dataset/nus-wide/setting-2/'
        train_set = my_NUS('train',  list_path , transform = transform )
        org_train_set = my_NUS('train', list_path , transform = val_transform )
        database_set = my_NUS('database', list_path, transform = val_transform)
        query_set = my_NUS('query', list_path, transform = val_transform)

        model_setting.update( load_setting('./settings/NUS-WIDE-2.txt') )
        
        label_embeddings_path = "./dataset/nus-wide/setting-2/core.txt"

        label_embeddings = load_label_embeddings(21,360,label_embeddings_path)
        model_setting['label_embeddings_path'] = label_embeddings_path
        label_embeddings = label_embeddings.to(device)

    elif ( ( dataset_name == 'ImageNet' ) and ( test_setting == '1' )):
        org_train_set,train_set , query_set , database_set = prepare_ImageNet(100 , transform , val_transform , True )
        
        model_setting.update( load_setting('./settings/ImageNet-1.txt') )

        label_embeddings = load_label_embeddings(100,360,"./dataset/ImageNet/core.txt")
        label_embeddings = label_embeddings.to(device)
    elif ( ( dataset_name == 'ImageNet' ) and ( test_setting == '2' )):
        org_train_set,train_set , query_set , database_set = prepare_ImageNet(100 , transform , val_transform , True )

        if ( backbone =='ResNet' ):
            model_setting.update( load_setting('./settings/ImageNet-2-ResNet.txt') )
        else:
            model_setting.update( load_setting('./settings/ImageNet-2-AlexNet.txt') )

        label_embeddings = load_label_embeddings(100,360,"./dataset/ImageNet/core.txt")
        label_embeddings = label_embeddings.to(device)
    else:
        print('Dataset Error')
        exit(1)
        

    core_rate_label_embeddings = model_setting['core_rate_label_embeddings']
    # core_norm = torch.norm( label_embeddings , dim=1 , p =2 )
    # label_embeddings.data /= core_norm.view(-1,1)
    label_embeddings*= core_rate_label_embeddings


    # x = get_cos_dist(label_embeddings, label_embeddings )
    # print('cos dist of label')
    # out_float_tensor(x)

    global tmp_train_datast
    tmp_train_datast = train_set

    global batch_size,len_code,n_class

    batch_size = model_setting['batch_size']

    len_code = model_setting['len_code']
    n_class = model_setting['n_class']

    model_setting['batch_query'] = 50

    train_loader = torch.utils.data.DataLoader(train_set, batch_size , shuffle=True, num_workers=4 )
    org_train_loader = torch.utils.data.DataLoader(org_train_set, batch_size , shuffle=False, num_workers=4 )
    query_loader = torch.utils.data.DataLoader(query_set, 50 , shuffle=False, num_workers = 4 )
    database_loader = torch.utils.data.DataLoader(database_set, 64 , shuffle=False, num_workers = 4 )

    n_train_set = train_set.__len__()
    n_query_set = query_set.__len__()
    n_database_set = database_set.__len__()

    model_setting['n_train_set'] = n_train_set
    model_setting['n_query_set'] = n_query_set
    model_setting['n_database_set'] = n_database_set

    list_dataloader = {}
    list_dataloader['train_loader'] = train_loader
    list_dataloader['query_loader'] = query_loader
    list_dataloader['database_loader'] = database_loader
    list_dataloader['org_train_loader'] = org_train_loader

    label_embeddings_norm = torch.norm( label_embeddings , p = 2, dim = 1 )
    print(' max = %.4f  min = %.4f '%( torch.max(label_embeddings_norm).item() , torch.min(label_embeddings_norm).item() ) )
    print(label_embeddings)

    global deep_quan,all_bits,n_codeword,len_subcode
    deep_quan = model_setting['deep_quan']
    all_bits = model_setting['all_bits']

    len_subcode = len_code // deep_quan

    model_setting['len_subcode'] =  len_subcode 
    model_setting['n_codeword'] =  2**(all_bits//deep_quan)
    model_setting['label_embeddings'] = label_embeddings

    if ( 'Train_core' in  model_setting ) and ( model_setting['Train_core'] ):
        global tmp_label_embeddings
        tmp_label_embeddings = torch.nn.Parameter(torch.rand(10, 360).to(device))
        # a = torch.nn.Parameter(torch.rand(10, 360).to(device))
        # a = a.to(device)
        # print(a.is_leaf )
        torch.nn.init.normal_(tmp_label_embeddings.data, std=1e-2)
        # print(tmp_label_embeddings.is_leaf )
        # exit()
        model_setting['label_embeddings'] = tmp_label_embeddings

    n_codeword = model_setting['n_codeword']


    global to_cls
    to_cls = torch.nn.Linear(len_code , n_class , bias = True).to(device)
    torch.nn.init.normal_(to_cls.weight.data,std=1e-2)
    torch.nn.init.constant_(to_cls.bias.data, 0.0)

    global fea_to_cls
    fea_to_cls = torch.nn.Linear(len_code , n_class , bias = True).to(device)
    torch.nn.init.normal_(fea_to_cls.weight.data,std=1e-2)
    torch.nn.init.constant_(fea_to_cls.bias.data, 0.0)
    model_setting['fea_to_cls'] = fea_to_cls

    return list_dataloader

def train(argv):
    start_train = time.time()
    model_setting = {}
    argv = argv[1:]
    for item in argv:
        a , b = item.split('=')
        model_setting[a] = b

    model_setting['device'] = torch.device(model_setting['device'])
    model_setting['test_setting'] = model_setting['test_setting']
    model_setting['deep_quan'] = int(model_setting['deep'])
    model_setting['all_bits'] = int(model_setting['all_bits'])
    model_setting.pop('deep')

    global device
    device = model_setting['device']

    list_dataloader = get_dataset(model_setting)

    get_priv_list(model_setting)

    print('model_setting = ',)
    for x in model_setting:
        print(x+' = '+str( model_setting[x] ))

    if ( model_setting['backbone'] == 'AlexNet'):
        # net  = model_feature(len_code,n_class , Is_normalize=False , Is_tanh=model_setting['Is_tanh'] ).to(device)
        net  = model_feature(len_code,n_class , Is_normalize=False , Is_tanh=True ).to(device)
        # net  = model_feature(len_code,n_class , Is_normalize=True , Is_tanh=True ).to(device)
        print('Is_normalize = ', net.Is_normalize)
    elif ( model_setting['backbone'] == 'ResNet'):
        net  = model_feature_resnet(len_code,n_class).to(device)
    else:
        print('backbone Error')
        exit()

    list_net = [net]
    for i in range(deep_quan):
        tmp_net = model_quantization(n_codeword, len_subcode  ).to(device)
        list_net.append(tmp_net)

    list_optimizer, list_scheduler, optimizer_fc, scheduler_fc = get_optimizers(list_net,model_setting)

    n_epoch_org = model_setting['n_epoch_org']
    n_epoch_code = model_setting['n_epoch_code']

    Load_flag = False
    if ( 'load_path' in model_setting ):
        Load_flag = True    
        open_path = model_setting['load_path']
        n_epoch_org = 0

    save_path = model_setting['save_path']

    print('save_path = ',save_path)

    label_embeddings = model_setting['label_embeddings']
    # for i in range(10):
    #     for j in range(10):
    #         ans = torch.cosine_similarity( label_embeddings[i], label_embeddings[j] , dim =0 ).item()
    #         ans = (ans+1)/2
    #         print('%.5f'%ans,end=' ')
    #     print('')
    # exit()

    # all_test = True
    all_test = False

    if ( all_test ):
        for i in range( len(list_net) ):
            checkpoint = torch.load( open_path+'_net_'+str(i) , map_location = device  )
            list_net[i].load_state_dict(checkpoint['model_state_dict'])
            print(' load from ', open_path+'_net_'+str(i) )
        list_net[0].eval()
        # get_cos_ap(list_dataloader,list_net, model_setting, False, 54000)
        test_mAP(list_dataloader,list_net, model_setting)
        exit()
        # test_mAP(list_dataloader,list_net, model_setting)
        
        # print('train_loader mse = ',test_mse( list_dataloader['train_loader'] , list_net, model_setting))
        # print('org_train_loader mse = ',test_mse( list_dataloader['org_train_loader'] , list_net, model_setting))
        # print('query_loader mse = ',test_mse( list_dataloader['query_loader'] , list_net, model_setting))
        # print('database_loader mse = ',test_mse( list_dataloader['database_loader'] , list_net, model_setting))
        # print('k_ = ',get_distinct_codewords(list_net))
        # cos_sim_core = get_cos_dist( label_embeddings , label_embeddings )
        # print(' cos_sim_core = ')
        # out_float_tensor(cos_sim_core)
        test_mAP_org(list_dataloader,net, model_setting)

        sim_arr ,un_sim_arr, mean_norm = test_gather(list_dataloader,list_net,model_setting)
        te_sim = torch.FloatTensor(sim_arr).to(device)
        te_un_sim = torch.FloatTensor(un_sim_arr).to(device)
        print('mean sim = ',te_sim.mean().item())
        print('mean unsim = ',te_un_sim.mean().item())

        print('mean norm sim = ',(te_sim.mean()/mean_norm).item())
        print('mean norm unsim = ',(te_un_sim.mean()/mean_norm).item())

        # M_r = test_knn(list_dataloader,list_net,model_setting)
        exit()
        # test_mAP(list_dataloader,list_net, model_setting)
        # get_tsne(list_dataloader,list_net,model_setting,is_org=True)

        # cnt_pic_code  = test_code_dis(list_dataloader,list_net, model_setting)
        # for i in range( deep_quan ):
        #     for j in range( n_codeword ):
        #         if ( cnt_pic_code[i][j] <=0 ):
        #             list_net[1+i].CodeBook[j].data *= 0
    
        # test_mAP(list_dataloader,list_net, model_setting)
        # cnt_pic_code = test_code_dis(list_dataloader,list_net, model_setting)
        exit(0)

    if ( Load_flag ):
        for i in range( 1 ):
        # for i in range( len(list_net) ):
            checkpoint = torch.load( open_path+'_net_'+str(i) , map_location = device  )
            list_net[i].load_state_dict(checkpoint['model_state_dict'])
            print(' load from ', open_path+'_net_'+str(i) )
        list_net[0].eval()
        test_mAP_org(list_dataloader,list_net[0], model_setting)
        # test_mAP(list_dataloader,list_net, model_setting)

        # cnt_pic_code = test_code_dis(list_dataloader,list_net, model_setting)

        # for i in range( deep_quan ):
        #     for j in range( n_codeword ):
        #         if ( cnt_pic_code[i][j] <=0 ):
        #             list_net[1+i].CodeBook[j].data *= 0
    
        # test_mAP(list_dataloader,list_net, model_setting)
        # cnt_pic_code = test_code_dis(list_dataloader,list_net, model_setting)
        # exit(0)
    
    for epoch in range( 1 , n_epoch_org+1 ):
        loss = opt_org(epoch,list_dataloader,list_net[0],list_optimizer,list_scheduler, model_setting)
        
        if ( epoch % 10 ==0 ):
        # if ( epoch % 1 ==0 ):
        # if ( ( ( epoch > 150 ) and ( epoch %20 == 0 ) ) or ( epoch % 50 ==0 ) ):
            for i in range( len( list_net ) ):
                torch.save({'model_state_dict': list_net[i] .state_dict()}, save_path+'_net_'+str(i) )
            print('save',loss)
            list_net[0].eval()
            test_mAP_org(list_dataloader,net, model_setting)
            list_net[0].train()
        # out_float_tensor(model_setting['label_embeddings'][0][:10])
        with torch.no_grad():
            list_net[0].eval()
            global tmp_train_datast
            tmp,_ = tmp_train_datast[0]
            tmp = tmp.unsqueeze(0)
            tmp = tmp.to(device)
            image_embeddings , _ = list_net[0](tmp)
            image_embeddings = image_embeddings[0,0:20]
            out_float_tensor(image_embeddings)
            list_net[0].train()
            

    for x in list_net[0].parameters():
        x.requires_grad = False
    
    # list_net[0].eval()
    # test_mAP(list_dataloader,list_net, model_setting)

    quan_f_eval = False
    if ( quan_f_eval == False ):
        list_net[0].train()
    else:
        list_net[0].eval()

    
    for epoch in range( 1 , n_epoch_code+1 ):
        loss = opt_code(epoch,list_dataloader,list_net,list_optimizer,list_scheduler, model_setting)

        if (epoch % 5 ==0 ):
            model_setting['soft_rate'] = max( model_setting['soft_rate']*model_setting['quan_soft_mul'] , 0.01 )
            model_setting['hard_rate'] = min( model_setting['hard_rate']*model_setting['quan_hard_mul'] , 30 )
            # model_setting['soft_rate'] = model_setting['soft_rate']*model_setting['quan_soft_mul']
            # model_setting['hard_rate'] = model_setting['hard_rate']*model_setting['quan_hard_mul']

        print(' soft = %.3f  hard= %.3f'%(model_setting['soft_rate'],model_setting['hard_rate']) )

        # if( ( epoch % 10 == 0 ) or ( epoch == 1 ) ):
        # if( epoch % 10 == 0 ):
        if ( ( (epoch >= 50 ) and ( epoch % 10 ==0 ) ) or ( epoch == 1 ) ):
            for i in range( len( list_net ) ):
                torch.save({'model_state_dict': list_net[i] .state_dict()}, save_path+'_net_'+str(i) )
            print('save',loss)

            list_net[0].eval()
            test_mAP(list_dataloader,list_net, model_setting)

            if ( quan_f_eval == False ):
                list_net[0].train()
            else:
                list_net[0].eval()
    time_elapsed = time.time() - start_train
    # print('Program finished in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Program finished in {:.0f}h {:.0f}m {:.0f}s'.format(time_elapsed//3600,time_elapsed%3600 // 60, time_elapsed % 60))
    list_net[0].eval()
    test_mAP(list_dataloader,list_net, model_setting)

if __name__ == '__main__':
    train(argv)