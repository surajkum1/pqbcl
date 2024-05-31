from utils import feature_loader
import numpy as np
from utils import Classifier,CustomDataset
import torch
from pqbcal import PQCLR
import pandas as pd

def streaming():

    train_features,train_paths = feature_loader('CORe50_resnet34_train_features.data','CORe50_resnet34_train_labels.data')
    print('Shape of train feature and train Paths is {} and {}'.format(train_features.shape,len(train_paths)))
    test_y = []
    test_x,test_paths = feature_loader('CORe50_resnet34_test_features.data','CORe50_resnet34_test_labels.data')
    for path in test_paths:
        cls_no = (int(path[5:7])-1)//5
        test_y.append(cls_no)
    test_y = np.array(test_y)
    print('Shape of test feature and test label is {} and {}'.format(test_x.shape,test_y.shape))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    no_test_samples = test_y.shape[0]
    test_x,test_y = torch.Tensor(test_x).to(device),torch.Tensor(test_y).to(device)

    test = {}
    for k in range(5):
        classifier = Classifier(10)
        classifier.to(device)
        model = PQCLR(num_classes = 10,device = 'cpu',dataset='CORe50',input_shape = 512,num_codebooks = 16,codebook_size = 256)
        opq,pq,index,labels_object= model.initialize('ImageNet_OQP_Num_cb_16_cb_size_256.pkl')
        model.device = device
        test_accuracies = []
        batch_size = 16
        max_count = 100
        for i in range(0,max_count):
            random_objects = model.data_processing.rand_data_extractor()
            data = model.active_sample_selection(random_objects,train_features,train_paths)
            features = data[0]
            labels = data[1]
            index,labels_object = model.encode(opq,index,labels_object,features,labels)
            custom_dataset = CustomDataset(range(index.ntotal))
            train_loader = torch.utils.data.DataLoader(custom_dataset, batch_size=batch_size,shuffle = True)
            criterion = torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(classifier.parameters(),lr=0.0001)
            with torch.enable_grad():
                for epoch in range(1,101):
                    correct_predictions = 0
                    total_samples = 0
                    running_loss = 0.0
                    for batch in train_loader:  # Iterate over batches of training data
                        inputs, labels = model.decode_batch(batch,opq,index,labels_object)
                        inputs, labels = inputs.to(device), labels.to(device)
                        # Forward pass

                        outputs = classifier(inputs.float())

                        # Compute loss
                        loss = criterion(outputs, labels.long())
                        # Backpropagation and optimization

                        optimizer.zero_grad()
                        loss.backward(retain_graph=True )
                        optimizer.step()

                        total_samples += labels.size(0)
                        running_loss += loss.item()/total_samples

                        # Compute accuracy for the current batch
                        _, predicted = torch.max(outputs, 1)  # Get the class with the highest probability
                        correct_predictions += (predicted == labels).sum().item()

                    # Calculate accuracy for the current epoch
                    accuracy = correct_predictions / total_samples

                    # Print accuracy for monitoring training progress
                    print(f'Epoch [{epoch}] -loss {running_loss:.4f} - Accuracy: {accuracy * 100:.2f}% ')
                    if accuracy >0.95:
                        break
        
            correct_predictions_test = 0
            with torch.no_grad():
                test_output = classifier(test_x)
                _, test_predicted = torch.max(test_output, 1)
                correct_predictions_test += (test_predicted == test_y).sum().item()
                test_accuracy = correct_predictions_test/no_test_samples
                test_accuracies.append(test_accuracy)
                print(f'\nExp_No : {k+1} - Test accuracy : {test_accuracy* 100:.2f}% -Current Object : {model.data_processing.selected_obj[-1]} - Object No- {i+1}')
        test[k+1] = test_accuracies
    return test

if __name__ == '__main__':
    test_accuracies = streaming()
    df = pd.DataFrame(test_accuracies)
    df.to_csv('CORe50_PQ_CS_adam_accuracies.csv')