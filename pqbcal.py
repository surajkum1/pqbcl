import torch
import numpy as np
from utils import cosine_similarity,distance,Labels
from data_processing import DataProcessing
import faiss
import pickle




class PQCLR:
    
    def __init__(self,num_classes = 10,device = 'cpu',dataset='CORe50',input_shape = 512,num_codebooks = 8,codebook_size = 256):
        np.random.seed(0)
        torch.manual_seed(0)
        self.data_means = [None]*num_classes
        self.weights = [None]*num_classes
        self.data_processing = DataProcessing(dataset=dataset)
        if device != "cpu":
            self.device = device
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.input_shape  = self.num_channels = input_shape
        self.codebook_size  = codebook_size
        self.num_codebooks = num_codebooks

    def initialize(self,OPQ_vector_file):
        with open(OPQ_vector_file, 'rb') as f:
            d = pickle.load(f)
        nbits = int(np.log2(self.codebook_size))
        pq = faiss.ProductQuantizer(self.num_channels, self.num_codebooks, nbits)
        opq = faiss.OPQMatrix(self.num_channels, self.num_codebooks)
        opq.pq = pq
        faiss.copy_array_to_vector(d['pq_centroids'].ravel(), pq.centroids)
        faiss.copy_array_to_vector(d['opq_A'].ravel(), opq.A)
        faiss.copy_array_to_vector(d['opq_b'].ravel(), opq.b)
        opq.is_trained = True
        opq.is_orthonormal = True
        index = faiss.IndexPQ(self.num_channels, self.num_codebooks, nbits)
        index.pq = pq
        index.is_trained = True
        labels_object = Labels()
        return opq,pq,index,labels_object
    
    def encode(self,opq,index,labels_object,features,labels):
        selected_obj_class = int(labels[0])
        if self.data_means[selected_obj_class] is not None:
            self.data_means[selected_obj_class].append(np.mean(features,axis=0))
        else:
            self.data_means[selected_obj_class] = [np.mean(features,axis=0)]
        features = opq.apply_py(np.ascontiguousarray(features, dtype=np.float32))
        index.add(np.ascontiguousarray(features, dtype=np.float32))
        labels_object.add(labels)
        assert index.ntotal == len(labels_object.labels),"Both sizes should be same"
        return index,labels_object
    
    def decode_batch(self,data_batch_index,opq,index,labels_object):
        x = []
        y = []
        for i in data_batch_index:
            
            recons_data = index.reconstruct(int(i))
            recons_data = opq.reverse_transform(recons_data.reshape(1,self.num_channels))
            x.append(recons_data.reshape(self.num_channels))
            y.append(labels_object.labels[i])
        return torch.Tensor(x).to(self.device),torch.Tensor(y).reshape(-1).to(self.device)

    # def update_knowledge_base(self,x,y,similarity_score_thresold = 0.9,
    #                         batch_size = 8,num_epochs =15,learning_rate = 5e-3,alpha=0.1): # x is data and y is class no.

    #     if self.data_means[y] is not None:
    #         similarity_scores = []
    #         curr_mean = np.mean(x,axis=0)
    #         for mean in self.data_means[y]:
    #             similarity_scores.append(cosine_similarity(curr_mean,mean))

    #         most_similar_index = similarity_scores.index(max(similarity_scores))
    #         if similarity_scores[most_similar_index]> similarity_score_thresold:
    #             with torch.no_grad():
    #                 x_rehearsal = self.sampling(y,most_similar_index)
    #                 x = torch.concatenate([torch.Tensor(x).to(self.device),x_rehearsal],axis=0)
    #                 self.data_means[y][most_similar_index]= torch.mean(x,axis=0).detach().cpu().numpy()
                
    #             with torch.enable_grad():
    #                 self.weights[y][most_similar_index] = x.shape[0]
    #                 decoder = vae_training(x,input_size = self.input_size,latent_size = self.latent_size,num_epochs=num_epochs,
    #                                        batch_size = batch_size,learning_rate = learning_rate,device= self.device,alpha = alpha)
    #                 self.vaes[y][most_similar_index] = decoder
    #         else:
    #             self.data_means[y].append(np.mean(x,axis=0))
    #             self.weights[y].append(x.shape[0])
    #             decoder = vae_training(x,input_size = self.input_size,latent_size = self.latent_size,num_epochs=num_epochs,
    #                         batch_size = batch_size,learning_rate = learning_rate,device= self.device,alpha = alpha)
    #             self.vaes[y].append(decoder)

    #     else:
    #         self.data_means[y] = [np.mean(x,axis=0)]
    #         self.weights[y] = [x.shape[0]]
    #         decoder = vae_training(x,input_size = self.input_size,latent_size = self.latent_size,num_epochs=num_epochs,
    #                    batch_size = batch_size,learning_rate = learning_rate,device= self.device,alpha = alpha)
    #         self.vaes[y] = [decoder]

    def active_sample_selection(self,random_objects,features,paths):
        best_cosine_distance = []
        datas = []

        for obj in random_objects:
            similarities = []
            if self.data_processing.dataset == 'CORe50':
                s,o = int(obj[:2]),int(obj[3:]) # s is season_no, o is object number and c is class
                data = self.data_processing.data_selector(features,paths,season = s, object = o)
                
            elif self.data_processing.dataset == 'cifar100':
                data = self.data_processing.data_selector(features,paths, object = obj)

            elif self.data_processing.dataset == 'cifar100_2':
                o,p = int(obj[1:3]),int(obj[-1]) # s is season_no, o is object number and c is class
                data = self.data_processing.data_selector(features,paths,object = o,part=p)
            
            
            data_mean = np.mean(data[0],axis=0)

            for cls_mean in self.data_means:
                if cls_mean is not None:
                    for mean in cls_mean:
                        similarities.append(cosine_similarity(mean,data_mean))

            if len(similarities)==0:
                self.data_processing.selected_obj.append(obj)
                return data

            best_cosine_distance.append(max(similarities))
            datas.append(data)

        index = best_cosine_distance.index(min(best_cosine_distance))
        self.data_processing.selected_obj.append(random_objects[index])
        return datas[index]

    

    