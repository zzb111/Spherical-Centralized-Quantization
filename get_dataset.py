import torch
from torchvision import datasets
from dataset import random_ImageNet , random_CIFAR10
import random
from tool import new_load_list_file,load_list_file, new_write_list_file, write_list_file

def prepare_ImageNet(use_class , transform , val_transform , flag_load_list):

    traindir = './data/ImageNet'

    base_train_set = datasets.ImageFolder(traindir,transform)
    base_database_set = datasets.ImageFolder(traindir,val_transform)

    list_class = []

    dir_list_train = './dataset/ImageNet/train.txt'
    dir_list_base = './dataset/ImageNet/database.txt'
    dir_list_query = './dataset/ImageNet/query.txt'
    dir_list_class = './dataset/ImageNet/class.txt'

    if ( flag_load_list ):
        list_data_train = new_load_list_file(dir_list_train)
        list_data_query = new_load_list_file(dir_list_query)
        list_data_base  = new_load_list_file(dir_list_base)

        list_class = load_list_file( dir_list_class )

        org_train_set = random_ImageNet( base_train_set ,   val_transform,  list_data=list_data_train )
        train_set = random_ImageNet( base_train_set ,       transform,      list_data=list_data_train )
        query_set = random_ImageNet( base_database_set ,    val_transform,  list_data=list_data_query )
        database_set = random_ImageNet( base_database_set , val_transform,  list_data=list_data_base  )

        print('load list OK')
    else:
        list_class = list ( range(1000) )
        random.shuffle(list_class)
        list_class = list_class[0:use_class]

        list_change_class = [1000]*1010

        for i in range( len(list_class) ):
            list_change_class[ list_class[i] ] = i

        list_label_ImageNet_train_st = []
        with open('./dataset/ImageNet/list_label_ImageNet_train_st.txt','r') as f:
            for line in f:
                words = line.split()
                for word in words:
                    list_label_ImageNet_train_st.append(int(word))
            f.close()
        
        list_label_ImageNet_val = []
        with open('./dataset/ImageNet/list_label_ImageNet_val.txt','r') as f:
            for line in f:
                word = line.split()
                list_label_ImageNet_val.append( (word[0], int(word[1])) )
            f.close()
        
        print('Load dataset list OK')

        list_sample = []

        cnt_train = [0]*use_class
        cnt_query = [0]*use_class
        cnt_base = [0]*use_class

        list_data_query = []
        list_data_train = []
        list_data_base = []

        for i in range(use_class):
            for j in range(list_label_ImageNet_train_st[ list_class[ i ] ] , list_label_ImageNet_train_st[ list_class[i] +1 ] ):
                list_sample.append( (0 , j , i  ) )

        random.shuffle(list_sample)
        for i in range( len( list_sample ) ):
            label = list_sample[i][2]
            if ( cnt_train[ label ]<100 ):
                cnt_train[ label ] += 1
                list_data_train.append( ( list_sample[i][0] , list_sample[i][1] , list_sample[i][2] ) )

            cnt_base[ label ] += 1
            list_data_base.append( ( list_sample[i][0] , list_sample[i][1] , list_sample[i][2] ) )

        list_sample = []
        for i in range( len(list_label_ImageNet_val) ):
            now_label = list_label_ImageNet_val[i][1]
            if ( list_change_class[ now_label ] < 100 ):
                list_sample.append( (1,'./data/ImageNet/val/'+list_label_ImageNet_val[i][0], list_change_class[ now_label ] ) )

        for i in range( len( list_sample ) ):
            label = list_sample[i][2]
            cnt_query[ label ] += 1
            list_data_query.append( ( list_sample[i][0] , list_sample[i][1] , list_sample[i][2] ) )
        
        print('Build sampels OK')

        new_write_list_file(list_data_train,    dir_list_train)
        new_write_list_file(list_data_query,    dir_list_query)
        new_write_list_file(list_data_base,     dir_list_base)
        write_list_file(list_class,         dir_list_class)

        org_train_set = random_ImageNet( base_train_set ,   val_transform,  list_data=list_data_train )
        train_set = random_ImageNet( base_train_set ,       transform,      list_data=list_data_train )
        query_set = random_ImageNet( base_database_set ,    val_transform,  list_data=list_data_query )
        database_set = random_ImageNet( base_database_set , val_transform,  list_data=list_data_base )

    print('first 10 classes = ',list_class[:10])
    return org_train_set,train_set, query_set, database_set


def prepare_CIFAR10( transform , val_transform, flag_load_list, list_path):

    train_path = list_path+'train.txt'
    query_path = list_path+'query.txt'
    database_path = list_path+'database.txt'

    if ( flag_load_list ):

        org_train_set = random_CIFAR10(  train_path , transform )
        train_set = random_CIFAR10(  train_path , transform )
        query_set = random_CIFAR10(  query_path , val_transform )
        database_set = random_CIFAR10( database_path , val_transform )

        print('load list OK')
    else:

        train_set = datasets.CIFAR10(root='./data/CIFAR-10/', train=True, download=False )
        test_set = datasets.CIFAR10(root='./data/CIFAR-10/', train=False, download=False )

        fp_train = open(train_path,"w")
        fp_query = open(query_path,"w")
        fp_base = open(database_path,"w")

        list_sample = []

        cnt_train = [0]*10
        cnt_query = [0]*10
        cnt_base = [0]*10

        list_sample = []
        cnt = 0
        for _,label in test_set:
            list_sample.append( (cnt,label) )
            cnt += 1

        for _,label in train_set:
            list_sample.append( (cnt,label) )
            cnt += 1

        random.shuffle(list_sample)
        for i in range( len( list_sample ) ):
            label = list_sample[i][1]
            id = list_sample[i][0]
            if ( cnt_train[ label ] < 500 ):
                cnt_train[ label ] += 1
                fp_train.write(  "./data/CIFAR-10/pics/" + str( id ) + ".jpg" + " "  + str(label) + '\n' )
            elif ( cnt_query[ label ] < 100 ):
                cnt_query[ label ] += 1
                fp_query.write(  "./data/CIFAR-10/pics/" + str( id ) + ".jpg" + " "  + str(label) + '\n' )
            else:
                cnt_base[ label ] += 1
                fp_base.write(  "./data/CIFAR-10/pics/" + str( id ) + ".jpg" + " "  + str(label) + '\n' )
        fp_train.close()
        fp_query.close()
        fp_base.close()

        print('Build sampels OK')
        
        org_train_set = random_CIFAR10(  train_path , transform )
        train_set = random_CIFAR10(  train_path , transform )
        query_set = random_CIFAR10( query_path , val_transform )
        database_set = random_CIFAR10(  database_path , val_transform )

    return org_train_set,train_set, query_set, database_set

def prepare_CIFAR10_datatrain( transform , val_transform, flag_load_list ,list_path):

    train_path = list_path+'train.txt'
    query_path = list_path+'query.txt'
    database_path = list_path+'database.txt'
    
    if ( flag_load_list ):
        org_train_set = random_CIFAR10(  train_path , transform )
        train_set = random_CIFAR10(  train_path , transform )
        query_set = random_CIFAR10(  query_path , val_transform )
        database_set = random_CIFAR10( database_path , val_transform )

        print('load list OK')
    else:
        print('write rand path = ',train_path)

        train_set = datasets.CIFAR10(root='./data/CIFAR-10/', train=True, download=False )
        test_set = datasets.CIFAR10(root='./data/CIFAR-10/', train=False, download=False )

        fp_train = open(train_path,"w")
        fp_query = open(query_path,"w")
        fp_base = open(database_path,"w")
        
        list_sample = []


        cnt_train = [0]*10
        cnt_query = [0]*10
        cnt_base = [0]*10
        
        list_sample = []
        cnt = 0
        for _,label in test_set:
            list_sample.append( (cnt,label) )
            cnt += 1

        for _,label in train_set:
            list_sample.append( (cnt,label) )
            cnt += 1

        random.shuffle(list_sample)
        for i in range( len( list_sample ) ):
            label = list_sample[i][1]
            id = list_sample[i][0]
            if ( cnt_train[ label ] < 5000 ):
                cnt_train[ label ] += 1
                fp_train.write(  "./data/CIFAR-10/pics/" + str( id ) + ".jpg" + " "  + str(label) + '\n' )

                cnt_base[ label ] += 1
                fp_base.write(  "./data/CIFAR-10/pics/" + str( id ) + ".jpg" + " "  + str(label) + '\n' )

            elif ( cnt_query[ label ] < 1000 ):
                cnt_query[ label ] += 1
                fp_query.write(  "./data/CIFAR-10/pics/" + str( id ) + ".jpg" + " "  + str(label) + '\n' )
            else:
                print('dataset Error')
                exit(0)

        fp_train.close()
        fp_query.close()
        fp_base.close()

        print('Build sampels OK')
        
        org_train_set = random_CIFAR10( train_path , transform )
        train_set = random_CIFAR10(  train_path , transform )
        query_set = random_CIFAR10(  query_path , val_transform )
        database_set = random_CIFAR10(  database_path , val_transform )

    return org_train_set,train_set, query_set, database_set