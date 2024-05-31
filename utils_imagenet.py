import os
import torch
from PIL import Image
import pickle
import numpy as np
from utils import Img2Vec
import faiss
import time

def feature_extrator(image_dir,dataset='Imagenet',model='resnet-18', layer='default', layer_output_size=512):
    img2vec = Img2Vec(cuda=True,model=model, layer=layer, layer_output_size=layer_output_size)
    image_features = []
    image_name = []
    None_type = []
    count = 0
    for classes in os.listdir(image_dir):
        for img_name in os.listdir(os.path.join(image_dir,classes)):
            # Read in an image (rgb format)
            img = Image.open(os.path.join(image_dir,classes,img_name))
            # Get a vector from img2vec, returned as a torch FloatTensor
            with torch.no_grad():
                vec = img2vec.get_vec(img)
            try:
                image_features.append(vec.flatten())
                image_name.append(img_name) 
            except  AttributeError:
                pass
            count += 1
            if count%10000 == 0:
                print(count)
    image_features = np.array(image_features)
    with open(dataset+"_"+model+'_features.pkl', 'wb') as filehandle:
        # store the data as binary data stream
        pickle.dump(image_features, filehandle)
    with open(dataset+'_labels.pkl', 'wb') as filehandle:
        # store the data as binary data stream
        pickle.dump(image_name, filehandle)
    with open(dataset+'_NoneType.pkl', 'wb') as filehandle:
        # store the data as binary data stream
        pickle.dump(None_type, filehandle)
    

def fit_opq(feats_base_init, num_channels,num_codebooks, codebook_size):
    start = time.time()
    train_data_base_init = np.reshape(feats_base_init, (-1, num_channels))
    nbits = int(np.log2(codebook_size))
    
    ## train opq
    print('Training Optimized Product Quantizer..')
    pq = faiss.ProductQuantizer(num_channels, num_codebooks, nbits) # 512, 8, 8
    opq = faiss.OPQMatrix(num_channels, num_codebooks) # 512, 8
    opq.pq = pq
    # opq.niter = 500 # optimum, higher than this provides same performance, default value=50
    #opq.verbose = True
    opq.train(np.ascontiguousarray(train_data_base_init, dtype=np.float32))
    train_data_base_init = opq.apply_py(np.ascontiguousarray(train_data_base_init, dtype=np.float32))
    pq.train(train_data_base_init)
    
    print(" OPQ training completed in {} secs".format(time.time() - start))
    del train_data_base_init
    ## get OPQ parameters
    d = num_channels
    A = faiss.vector_to_array(opq.A).reshape(d, d) ## from Swig Object to np array
    b = faiss.vector_to_array(opq.b) ## from Swig Object to np array

    ## get PQ centroids/codebooks
    centroids = faiss.vector_to_array(pq.centroids).reshape(pq.M, pq.ksub, pq.dsub)
    ## save in a dictionary
    d = { 'opq_A': A, 'opq_b': b, 'pq_centroids': centroids}
    with open(f'ImageNet_OQP_Num_cb_{num_codebooks}_cb_size_{codebook_size}.pkl', 'wb') as filehandle:
        # store the data as binary data stream
        pickle.dump(d, filehandle)
    return pq, opq