from PIL import Image
import torch


class random_CIFAR10(torch.utils.data.Dataset):
    def __init__(self,  list_path  , transform=None ): 
        super(random_CIFAR10,self).__init__()

        fp = open( list_path, "r")
        print(fp.name)
        list_image = []

        cnt = 0
        for line in fp:
            words = line.split()
            list_image.append((words[0],int(words[1])))
            cnt +=1
        fp.close()

        self.list_image  = list_image
        self.transform = transform

    def __getitem__(self, index):    

        fn, label = self.list_image[index] 
        img = Image.open( fn ).convert('RGB')
        
        if self.transform is not None:
            img = self.transform(img)
        return img , label

    def __len__(self): 
        return len( self.list_image )

class random_ImageNet(torch.utils.data.Dataset):
    def __init__(self, base_dataset , transform=None , list_data = [] ):
        super(random_ImageNet,self).__init__()
        self.base_dataset = base_dataset
        self.list_data  = list_data
        self.transform = transform
        self.cnt = [0]*100

    def __getitem__(self, index):
        id_dataset = self.list_data[index][0]
        id_pic = self.list_data[index][1]
        label = self.list_data[index][2]

        if ( id_dataset == 0 ):
            img , _ = self.base_dataset[ int( id_pic ) ]
            # print('')
        else:
            fn = id_pic
            img = Image.open( fn ).convert('RGB')
            self.cnt[label] += 1

            img = self.transform(img)

        return img , label

    def __len__(self):
        return len( self.list_data )

class my_NUS(torch.utils.data.Dataset):
    def __init__(self, set_type, list_path , transform=None):
        super(my_NUS,self).__init__()
        self.set_type = set_type
        if ( set_type == 'query' ):
            fp = open( list_path + "query.txt", "r")
        else:
            if ( set_type == 'train' ):
                fp = open( list_path + "train.txt", "r")
            else:
                if ( set_type == 'database' ):
                    fp = open( list_path + "database.txt", "r")
                else:
                    print('Dataset Type Error')
                    exit()
        print(fp.name)
        list_image = []
        for line in fp:
            words = line.split()
            tmp = torch.FloatTensor( 21 )
            for i in range(21):
                if ( words[i+1] == '1' ):
                    tmp[i] = 1
                else:
                    tmp[i] = 0
            list_image.append( ( words[0] , tmp ) )

        fp.close()

        self.list_image  = list_image
        self.transform = transform

    def __getitem__(self, index):    

        fn, label = self.list_image[index] 
        img = Image.open( fn ).convert('RGB')
        
        if self.transform is not None:
            img = self.transform(img)
        return img , label

    def __len__(self): 
        return len( self.list_image )