import torch
import torch.nn as nn
from torchvision import models
from func_dist import get_L1_dist , get_L2_dist

class model_feature(nn.Module):
    def __init__(self, n_features, n_labels, Is_normalize,Is_tanh):
        super(model_feature, self).__init__()
        self.Is_normalize = Is_normalize
        self.Is_tanh = Is_tanh
        model = models.alexnet(pretrained=True)

        fc1 = model.classifier[1]

        fc2 = model.classifier[4]

        self.mid = nn.Sequential(
            nn.Dropout(p=0.5, inplace=False),
            fc1,
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            fc2,
            nn.ReLU(inplace=True),
        )

        self.features = model.features
        self.avgpool = model.avgpool

        self.extractor = nn.Linear(4096, n_features)
        torch.nn.init.normal_(self.extractor.weight.data, std=1e-2)
        torch.nn.init.constant_(self.extractor.bias.data, 0.0)

        self.tanh = nn.Tanh()

        self.classifier = nn.Linear(4096, n_labels)
        torch.nn.init.normal_(self.classifier.weight.data, std=1e-2)
        torch.nn.init.constant_(self.classifier.bias.data, 0.0)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.mid(x)
        out_predict = self.classifier(x)

        if ( self.Is_tanh ):
            tmp = self.tanh(self.extractor(x))
        else:
            tmp = self.extractor(x)

        if ( self.Is_normalize ):
            out_feature = tmp/torch.norm(tmp , dim = 1 , p = 2 ).view(-1,1)*10
        else:
            out_feature = tmp
    
        return out_feature, out_predict


class model_feature_resnet(nn.Module):
    def __init__(self, n_features, n_labels):
        super(model_feature_resnet, self).__init__()

        model = models.resnet50(pretrained=True)

        self.features = nn.Sequential(model.conv1,
                                      model.bn1,
                                      model.relu,
                                      model.maxpool,
                                      model.layer1,
                                      model.layer2,
                                      model.layer3,
                                      model.layer4)

        self.avgpool = model.avgpool

        self.extractor = nn.Linear(2048, n_features)
        torch.nn.init.normal_(self.extractor.weight.data, std=1e-2)
        torch.nn.init.constant_(self.extractor.bias.data, 0.0)

        self.tttanh = nn.Tanh()

        self.classifier = nn.Linear(2048, n_labels)
        torch.nn.init.normal_(self.classifier.weight.data, std=1e-2)
        torch.nn.init.constant_(self.classifier.bias.data, 0.0)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        out_feature = self.tttanh(self.extractor(x))
        out_predict = self.classifier(x)
        return out_feature, out_predict


class model_quantization(nn.Module):
    def __init__(self, n_codeword, len_vec):
        super(model_quantization, self).__init__()
        self.CodeBook = torch.nn.Parameter(torch.rand(n_codeword, len_vec))
        torch.nn.init.normal_(self.CodeBook.data, std=1e-4)

    def forward(self, x , mask=None):
            

        dist = -get_L2_dist(x, self.CodeBook )

        if mask != None:
            dist += mask

        MAX_id = torch.argmax(dist, 1)
        Q_hard = torch.index_select(self.CodeBook, 0, MAX_id)
        return Q_hard , MAX_id
