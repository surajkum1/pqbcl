import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import pickle
from numpy.linalg import norm


class Img2Vec():

    def __init__(self, cuda=False, model='resnet-18', layer='default', layer_output_size=512):
        """ Img2Vec
        :param cuda: If set to True, will run forward pass on GPU
        :param model: String name of requested model
        :param layer: String or Int depending on model.  See more docs: https://github.com/christiansafka/img2vec.git
        :param layer_output_size: Int depicting the output size of the requested layer
        """
        self.device = torch.device("cuda" if cuda else "cpu")
        self.layer_output_size = layer_output_size
        self.model_name = model

        self.model, self.extraction_layer = self._get_model_and_layer(model, layer)

        self.model = self.model.to(self.device)

        self.model.eval()

        #self.scaler = transforms.Scale((224, 224))
        #self.scaler = transforms.Resize((224, 224))
        self.resizer = transforms.Resize(256)
        self.scaler = transforms.CenterCrop(224)
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])
        self.to_tensor = transforms.ToTensor()

    def get_vec(self, img, tensor=False):
        """ Get vector embedding from PIL image
        :param img: PIL Image or list of PIL Images
        :param tensor: If True, get_vec will return a FloatTensor instead of Numpy array
        :returns: Numpy ndarray
        """
        if type(img) == list:
            a = [self.normalize(self.to_tensor(self.scaler(im))) for im in img]
            images = torch.stack(a).to(self.device)
            if self.model_name == 'alexnet':
                my_embedding = torch.zeros(len(img), self.layer_output_size)
            else:
                my_embedding = torch.zeros(len(img), self.layer_output_size, 1, 1)

            def copy_data(m, i, o):
                my_embedding.copy_(o.data)

            h = self.extraction_layer.register_forward_hook(copy_data)
            h_x = self.model(images)
            h.remove()

            if tensor:
                return my_embedding
            else:
                if self.model_name == 'alexnet':
                    return my_embedding.numpy()[:, :]
                else:
                    print(my_embedding.numpy()[:, :, 0, 0].shape)
                    return my_embedding.numpy()[:, :, 0, 0]
        else:
            #image = self.normalize(self.to_tensor(self.scaler(img))).unsqueeze(0).to(self.device)
            # applying transforms
            img = self.resizer(img)
            img = self.scaler(img)
            img = self.to_tensor(img)
            if len(img)==3:
                img = self.normalize(img)
                image = img.unsqueeze(0).to(self.device)

                if self.model_name == 'alexnet':
                    my_embedding = torch.zeros(1, self.layer_output_size)
                else:
                    my_embedding = torch.zeros(1, self.layer_output_size, 1, 1)

                def copy_data(m, i, o):
                    my_embedding.copy_(o.data)

                h = self.extraction_layer.register_forward_hook(copy_data)
                h_x = self.model(image)
                h.remove()

                if tensor:
                    return my_embedding
                else:
                    if self.model_name == 'alexnet':
                        return my_embedding.numpy()[0, :]
                    else:
                        return my_embedding.numpy()[0, :, 0, 0]
            else:
                return None

    def _get_model_and_layer(self, model_name, layer):
        """ Internal method for getting layer from model
        :param model_name: model name such as 'resnet-18'
        :param layer: layer as a string for resnet-18 or int for alexnet
        :returns: pytorch model, selected layer
        """
        if model_name == 'resnet-18':
            model = models.resnet18(pretrained=True)
            if layer == 'default':
                layer = model._modules.get('avgpool')
                self.layer_output_size = 512
            else:
                layer = model._modules.get(layer)

            return model, layer

        elif model_name == 'resnet-34':
            model = models.resnet34(pretrained=True)
            if layer == 'default':
                layer = model._modules.get('avgpool')
                self.layer_output_size = 512
            else:
                layer = model._modules.get(layer)
            return model, layer

        elif model_name == 'alexnet':
            model = models.alexnet(pretrained=True)
            if layer == 'default':
                layer = model.classifier[-2]
                self.layer_output_size = 4096
            else:
                layer = model.classifier[-layer]

            return model, layer

        else:
            raise KeyError('Model %s was not found' % model_name)


class Labels:
    def __init__(self):
        self.labels = None
    def add(self,labels):
        if self.labels is not None:
            self.labels = np.concatenate((self.labels,labels),axis=0)
        else:
            self.labels = labels

def feature_loader(path_feature,path_label):
  flatten_features = []
  with open(path_feature, 'rb') as filehandle:
    # read the data as binary data stream
    features = pickle.load(filehandle)
  with open(path_label, 'rb') as filehandle:
    # read the data as binary data stream
    paths = pickle.load(filehandle)
  for i in features:
    flatten_features.append(np.array(i).astype(np.float32))
  return np.array(flatten_features),paths

class Classifier(nn.Module):

    def __init__(self,num_class):
        super(Classifier,self).__init__()

        self.linear1 = nn.Linear(512,num_class)

    def forward(self,x):
        x = self.linear1(x)
        return x

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        # self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        # label = self.labels[index]

        if self.transform:
            sample = self.transform(sample)

        return sample#, label

def distance(a,b):
    return norm(a-b)

def cosine_similarity(A,B):
    return A.dot(B)/(norm(A)*norm(B))