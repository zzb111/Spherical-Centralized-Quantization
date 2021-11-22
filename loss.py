from tool import out_float_tensor
from scipy.ndimage.measurements import label
import torch
import torch.nn.functional as F
from func_dist import get_L2_dist,get_cos_dist,get_MSE_dist , get_L1_dist


def KL_reg_loss(dist,mask):
    # reg_sum = 0.0
    # for i in range( dist.size(0) ):
    #     logit = torch.nn.functional.softmax(dist[i],dim=0)
    #     prob = torch.distributions.categorical.Categorical( logits=logit  )
    #     prior = torch.distributions.categorical.Categorical( logits=torch.nn.functional.softmax( mask[i],dim=0 ) )
    #     reg = torch.distributions.kl_divergence( prob , prior)
    #     reg_sum += reg
    # print(reg_sum)
    # print( torch.nn.functional.softmax(dist,dim=1)[0] )
    # print( torch.nn.functional.softmax(mask,dim=1)[0] )
    prob = torch.distributions.categorical.Categorical( logits=torch.nn.functional.softmax(dist,dim=1)  )
    prior = torch.distributions.categorical.Categorical( logits=torch.nn.functional.softmax(mask,dim=1 ) )
    reg = torch.distributions.kl_divergence( prob , prior).mean()
    
    # reg_sum -= reg
    # print(reg_sum)
    # print(reg)
    # exit(0)
    return reg


def MAX_MIN_regular_loss(list_net,model_setting):
    cnt = 0
    regs = 0.0
    priv_list = model_setting['priv_list']
    
    if ( 0 == priv_list[-1][1]):
        for deep in range( 1, len(list_net) ):
            CodeBook = list_net[deep].CodeBook
            dist_mask = torch.eye( CodeBook.size(0) ).to(model_setting['device'])*1e9
            dist = get_L2_dist(CodeBook[:] , CodeBook[:]) + dist_mask
            MIN_id = torch.argmin(dist, 1)
            MIN_dist = torch.gather( dist, 1, MIN_id.view(-1,1))
            regs += MIN_dist.min()
            cnt += 1
        regs /= cnt
        return -torch.tanh(regs)

    for deep in range( 1, len(list_net) ):
        CodeBook = list_net[deep].CodeBook
        for st,ed in priv_list:
            dist_mask = torch.eye( ed-st).to(model_setting['device'])*1e9
            dist = get_L2_dist(CodeBook[st:ed] , CodeBook[st:ed]) + dist_mask
            MIN_id = torch.argmin(dist, 1)
            MIN_dist = torch.gather( dist, 1, MIN_id.view(-1,1))
            regs += MIN_dist.min()
            cnt += 1
    if ( cnt >0 ):
        regs /= cnt
    return -torch.tanh(regs)


def classify_loss(predicts,labels):
    return F.cross_entropy(predicts,labels)

def MSE_loss( image_embeddings , rebuild_vec , device ):
            
    delta = ( image_embeddings - rebuild_vec )
    mean_mse = torch.sum(  delta*delta ) / image_embeddings.size(0)
    return mean_mse

def cls_quan_loss( labels , rebuild_vec , to_cls, device ):
    predicts = to_cls(rebuild_vec)
    return F.cross_entropy(predicts,labels)


def CE_loss( predicts : torch.FloatTensor , labels : torch.LongTensor,  model_setting ):
    Is_single_label = model_setting['Is_single_label']
    if ( Is_single_label ):
        CE = classify_loss( predicts , labels )
    else:
        dist = torch.softmax( predicts , dim = 1)
        CE = - torch.mean( torch.log( torch.sum( dist * labels , dim = 1 ) ) )
    return CE

def get_core( labels: torch.FloatTensor , label_embeddings : torch.FloatTensor ):
    # return torch.mm( labels , label_embeddings  )/ torch.sum( labels , dim =1  ).view(-1,1)
    tmp =  torch.mm( labels , label_embeddings  )
    tmp_norm = torch.norm( tmp , p=2 , dim =1).view(-1,1)
    target = tmp/tmp_norm*torch.norm(label_embeddings[0],p=2,dim=0)
    return target

# def core_loss_pre( image_embeddings , labels , label_embeddings ,  model_setting ):
#     device = model_setting['device']
#     Is_single_label = model_setting['Is_single_label']

#     core_mse_rate = model_setting['core_mse_rate']

#     n_images = image_embeddings.size(0)

#     core_loss = torch.zeros(1).to(device)
    
#     mean_core_like = torch.zeros(1).to(device)
#     mean_core_like.requires_grad = False
    
#     if ( Is_single_label == False):
#         cores = get_core( labels , label_embeddings )

#     for n in range(n_images):
#         aim_embedding = torch.zeros_like( image_embeddings[0] )
        
#         if ( Is_single_label ):
#             i = labels[n].item()
#             aim_embedding = label_embeddings[i]
#         else:
#             aim_embedding = cores[n]

#         core_like = torch.cosine_similarity( aim_embedding , image_embeddings[n] , dim = 0)
#         core_like = 0.5+ core_like/2
#         mean_core_like += core_like
#         core_loss += 1-core_like
#         core_loss += core_mse_rate*torch.norm( aim_embedding - image_embeddings[n], p = 2, dim = 0 )

#     mean_core_like /= n_images
#     core_loss = core_loss  / n_images

#     return core_loss ,  mean_core_like

def core_loss( image_embeddings , labels , label_embeddings ,  model_setting ):
    device = model_setting['device']
    Is_single_label = model_setting['Is_single_label']
    core_norm_reg_rate = model_setting['core_norm_reg_rate']
    core_rate_label_embeddings = model_setting['core_rate_label_embeddings']

    core_mse_rate = model_setting['core_mse_rate']

    n_images = image_embeddings.size(0)

    core_loss = torch.zeros(1).to(device)
    
    mean_core_error = torch.zeros(1).to(device)
    mean_core_error.requires_grad = False
    
    if ( Is_single_label == False):
        cores = get_core( labels , label_embeddings )

    for n in range(n_images):
        aim_embedding = torch.zeros_like( image_embeddings[0] )
        
        if ( Is_single_label ):
            i = labels[n].item()
            aim_embedding = label_embeddings[i]
        else:
            aim_embedding = cores[n]
        now_norm = torch.norm(image_embeddings[n],p=2,dim=0 )
        core_loss += ( - torch.cosine_similarity( aim_embedding , image_embeddings[n] , dim=0 ) + 
        core_mse_rate*torch.norm( image_embeddings[n]-aim_embedding , p=2 , dim = 0 ) +
        core_norm_reg_rate*( ((now_norm - core_rate_label_embeddings )/ core_rate_label_embeddings)**2 )
        )



        mean_core_error += torch.cosine_similarity( aim_embedding , image_embeddings[n] , dim=0 )

    mean_core_error /= n_images
    core_loss = core_loss  / n_images

    return core_loss ,  mean_core_error


def part_loss(part_embeddings,quan_net,st_code ,ed_code,all_dist , mask ,  model_setting ):

    device = model_setting['device']
    soft_loss_rate = model_setting['soft_loss_rate']

    soft_rate = model_setting['soft_rate']
    hard_rate = model_setting['hard_rate']
    # quan_rate_gap = model_setting['quan_rate_gap']
    
    if ( st_code == ed_code ):
        a = torch.zeros(1).to(device)
        b = torch.zeros(1).to(device)
        return  a,b

    n_images = part_embeddings.size(0)

    org_dist =  all_dist[ : , st_code:ed_code ]

    neglect_dist_max = torch.ones_like(org_dist)*(1e15)

    neglect_dist_min = torch.ones_like(org_dist)*(-1e15)

    mask_dist_max = torch.where( mask>0 , org_dist , neglect_dist_max )
    mask_dist_min = torch.where( mask>0 , org_dist , neglect_dist_min )

    dist_max , _ = torch.max( mask_dist_min , dim = 1 )

    dist_max_fix = torch.zeros( n_images ).to(device)
    dist_max_fix.requires_grad = False

    dist_max_fix = dist_max.data
    
    norm_dist = mask_dist_max / dist_max.view(-1,1)

    dist = - norm_dist

    soft_dist = torch.zeros_like(dist)
    soft_dist.requires_grad = False
    soft_dist += torch.softmax( soft_rate*dist , dim = 1 )


    hard_dist = torch.zeros_like(dist)
    hard_dist.requires_grad = False
    hard_dist += torch.softmax( hard_rate*dist , dim = 1 )

    # if ( hard_rate > 2 ):
    #     print('hard =')
    #     out_float_tensor(hard_dist)
    #     print('')
    #     print('soft = ')
    #     out_float_tensor(soft_dist)
    #     exit()

    soft_quan = torch.mm( soft_dist.view(n_images,-1) , quan_net.CodeBook[ st_code:ed_code , : ] ).view(n_images,-1)
    hard_quan = torch.mm( hard_dist.view(n_images,-1) , quan_net.CodeBook[ st_code:ed_code , : ] ).view(n_images,-1)
    

    tmp_loss = soft_loss_rate*get_MSE_dist( part_embeddings , soft_quan ) + get_MSE_dist( part_embeddings , hard_quan )
    # tmp_loss = ( soft_loss_rate*get_MSE_dist( part_embeddings , soft_quan ) + get_MSE_dist( part_embeddings , hard_quan ) + 
    # quan_rate_gap* get_MSE_dist( soft_quan , hard_quan) )
    
    tmp_cos_dist , _ = torch.max( (get_cos_dist( part_embeddings , quan_net.CodeBook[ st_code:ed_code , : ] )/2 + 0.5)*mask , dim = 1 )
    return tmp_loss.mean() , tmp_cos_dist



def p_priv_loss(st_code ,ed_code,all_dist , mask_pri_one_zero, model_setting ):
    device = model_setting['device']
    
    if ( st_code == ed_code ):
        a = torch.zeros(1).to(device)
        b = torch.zeros(1).to(device)
        return  a

    n_images = mask_pri_one_zero.size(0)
    org_dist =  all_dist[ : , st_code:ed_code ]

    dist_max , _ = torch.max( org_dist , dim = 1 )

    dist_max_fix = torch.zeros( n_images ).to(device)
    dist_max_fix.requires_grad = False

    dist_max_fix = dist_max.data
    
    norm_dist = org_dist / dist_max.view(-1,1)
    
    dist = - org_dist

    # print( dist )
    # exit(0)
    p = torch.softmax( dist , dim = 1 )
    priv_p = -torch.sum( p*mask_pri_one_zero , dim = 1 )
    # print( priv_p )
    # exit(0)
    return priv_p.mean()

def new_MSE_loss( image_embeddings , labels , list_net ,  model_setting ):

    n_class = model_setting['n_class']
    len_subcode = model_setting['len_subcode']
    deep_quan = model_setting['deep_quan']
    n_codeword = model_setting['n_codeword']
    device = model_setting['device']
    Is_single_label = model_setting['Is_single_label']

    priv_list = model_setting['priv_list']
    
    n_image = image_embeddings.size(0)
    
    sum_MSE_loss = torch.zeros(1).to(device)

    mean_dist_1 = torch.zeros(1).to(device)
    mean_dist_2 = torch.zeros(1).to(device)
    mean_dist_3 = torch.zeros(1).to(device)

    mean_priv_loss = 0.0
    mean_public_loss = 0.0
    mean_push_loss = 0.0
    mean_select_priv_loss = 0.0
    
    mask_priv = torch.ones( n_image, priv_list[-1][1] ).to(device)
    mask_priv.requires_grad = False
    mask_priv = mask_priv*(-1e15)

    if ( Is_single_label ):
        for i in range( n_image ):
            mask_priv [  i , priv_list[ labels[i] ][0] : priv_list[ labels[i] ][1] ] = 1 
    else:
        for i in range( n_image ):
            for j in range( n_class ):
                if ( labels[i][j] == 1 ):
                    mask_priv [  i , priv_list[j][0] : priv_list[j][1] ] = 1

    mask_pri_one_zero = torch.clamp( mask_priv , min = 0 , max = 1 )
    # print(mask_pri_one_zero[0])
    # exit()

    mask_push = torch.ones( n_image, priv_list[-1][1] ).to(device)
    mask_push.requires_grad = False

    if ( Is_single_label ):
        for i in range( n_image ):
            mask_push [  i , priv_list[ labels[i] ][0] : priv_list[ labels[i] ][1] ] = -1e15
    else:
        for i in range( n_image ):
            for j in range( n_class ):
                if ( labels[i][j] == 1 ):
                    mask_push [  i , priv_list[j][0] : priv_list[j][1] ] = -1e15

    mask_public = torch.ones( n_image, n_codeword - priv_list[-1][1] ).to(device)
    mask_public.requires_grad = False

    for j in range(  deep_quan ):
        part_embeddings = image_embeddings[ : , j*len_subcode : (j+1)*len_subcode ] 
        
        all_dist = get_L2_dist( part_embeddings , list_net[j+1].CodeBook  )

        priv_loss , priv_dist = part_loss( part_embeddings ,list_net[j+1], priv_list[0][0] , priv_list[-1][1] ,all_dist,mask_priv ,  model_setting )
        mean_dist_1 += priv_dist.mean().item()

        public_loss , public_dist = part_loss( part_embeddings ,list_net[j+1], priv_list[-1][1] ,n_codeword,all_dist,mask_public ,  model_setting )
        mean_dist_2 += public_dist.mean().item()

        # push_priv_loss , push_dist = part_loss( part_embeddings ,list_net[j+1], priv_list[0][0] , priv_list[-1][1] ,all_dist,mask_push ,  model_setting )
        # mean_dist_3 += push_dist.mean().item()
        select_priv_loss = 0


        sum_MSE_loss += priv_loss + public_loss

        mean_priv_loss += priv_loss
        mean_public_loss += public_loss
        # mean_push_loss += push_priv_loss
        mean_select_priv_loss += select_priv_loss
        
    return sum_MSE_loss , mean_dist_1 , mean_dist_2 , mean_dist_3 , mean_priv_loss , mean_public_loss , mean_push_loss, mean_select_priv_loss

def adaptive_margin_loss( image_embeddings , labels , label_embeddings ,  model_setting ):
    device = model_setting['device']

    n_images = image_embeddings.size(0)

    core_loss = torch.zeros(1).to(device)
    n_class = label_embeddings.size(0)
        
    mean_core_error = torch.zeros(1).to(device)
    mean_core_error.requires_grad = False
    std_gap = torch.zeros(n_class,n_class).to(device)
    for i in range(n_class):
        std_gap[i] = 1- torch.cosine_similarity(label_embeddings[i].view(1,-1), label_embeddings ,dim=1 )
    if ( model_setting['Is_single_label']):
        for n in range(n_images):
            label = labels[n]
            cos = torch.cosine_similarity(image_embeddings[n].view(1,-1),label_embeddings,dim=1)
            for i in range( n_class ):
                if ( i!=label) :
                    core_loss += max( std_gap[i][label] - cos[label] + cos[i] , 0 )
    else:
        for n in range(n_images):
            cos = torch.cosine_similarity(image_embeddings[n].view(1,-1),label_embeddings,dim=1)
            # tmp = 0.0
            # for i in range(n_class):
            #     if ( labels[n][i] == 1):
            #         for j in range( n_class ):
            #             if ( labels[n][j] == 0):
            #                 core_loss += max( std_gap[i][j] - cos[i] + cos[j] , 0 )
            #                 tmp += max( std_gap[i][j] - cos[i] + cos[j] , 0 )
            diff = torch.clamp_min( std_gap - cos.view(-1,1) + cos.view(1,-1) , 0 )
            mask = torch.mm( labels[n].view(-1,1), 1-labels[n].view(1,-1) )
            mine = torch.sum( diff * mask )
            core_loss += mine
            # if ( torch.abs( mine - tmp ) > 1e-4 ):
            #     print('fail ')
            #     print('mine = ',mine.item())
            #     print('tmp = ', tmp.item())
            #     exit()
            
            
    core_loss = core_loss  / n_images  / ( n_class-1)
    return core_loss 