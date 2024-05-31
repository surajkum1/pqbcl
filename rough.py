from utils_imagenet import fit_opq
import logging
import pickle

if __name__ == "__main__":
    logger = logging.getLogger()
    
    with open('Imagenet_resnet-34_features.pkl', 'rb') as filehandle:
        # store the data as binary data stream
         data =pickle.load(filehandle)
    fit_opq(data, 512, 16, 256)