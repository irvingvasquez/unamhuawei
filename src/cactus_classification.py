'''
Exploratory data analisys for cactus classification
@juan1rving
'''

import os
import argparse

import numpy as np
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import optim
from torchvision import datasets, transforms
from torch.autograd import Variable

class LeNet5(nn.Module):
    def __init__(self, n_clases):
        '''
        Construimos la estructura de LeNet5
        
        '''
        super(LeNet5, self).__init__()
        
        # De acuerdo al artículo de LeCun La primera capa está compuesta por 6 kernels de 5x5
        self.conv1 = nn.Conv2d(3, 6, 5) # 1 canal de entrada 6 feature maps de salida, kernels de 5x5
        
        # Después tenemos una capa maxpooling
        # kernel_size=2, stride=2
        self.pool1 = nn.MaxPool2d(2, 2)
        
        # Agregamos otra capa convolucional con 16 kernels de 5 x 5
        self.conv2 = nn.Conv2d(6, 16, 5)
        
        # Maxpooling
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # Capas totalmente conectadas
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, n_clases)
    
    def forward(self, x):
        '''
        Definimos el pase frontal (forward pass)
        '''
        # Agregamos los ReLUs
        x = self.pool1(F.relu(self.conv1(x)))
        #print(x.size())
        x = self.pool2(F.relu(self.conv2(x)))

        # prep for linear layer by flattening the feature maps into feature vectors
        x = x.view(x.size(0), -1)
        # capas lineales
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return F.log_softmax(x, dim=1)


# Implementamos una función de evaluación
def validation(model, testloader, criterion, device):
    test_loss = 0
    accuracy = 0
    for images, labels in testloader:

        # warp input images in a Variable wrapper
        images = Variable(images)
        images = images.to(device)
        
        labels = Variable(labels)
        labels = labels.to(device)
        
        output = model.forward(images)
        
        test_loss += criterion(output, labels).item()
        
        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()

    return test_loss, accuracy


def main(folder, device, batch_size, epochs=2, learning_rate = 0.001):
    # directorio de la carpeta donde esta el dataset
    directorio = 'cactus_dataset2'
    directorio = folder

    # Conjunto de entrenamiento

    # aplicaré una serie de transformaciones
    # 1. escalar las imágenes a 32 x 32 pixeles
    # 2. recortar a 32x32
    # Aumentación de datos:
    # 3. Espejo horizontal con probabilidad de ser aplicado igual a p
    # 4. Espejo vertical con probabilidad igual a p
    # 5. convertir a tensores
    # 6. Normalizar
    transformaciones_training = transforms.Compose([transforms.Resize(32),
                                transforms.CenterCrop(32),
                                transforms.RandomHorizontalFlip(p=0.5),
                                transforms.RandomVerticalFlip(p=0.5),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                ])

    # crear el objeto cargador de datos
    trainset = datasets.ImageFolder(directorio + '/training_set', transformaciones_training) 
    trainloader = torch.utils.data.DataLoader(trainset, batch_size, shuffle=True)

    # Conjunto de validacíon
    # aplicaré una serie de transformaciones
    # 1. escalar las imágenes a 32 x 32 pixeles
    # 2. convertir a tensores
    transformaciones_validacion = transforms.Compose([transforms.Resize(32),
                                transforms.CenterCrop(32),
                                transforms.ToTensor(), 
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


    testset = datasets.ImageFolder(directorio + '/validation_set', transform=transformaciones_validacion)
    testloader = torch.utils.data.DataLoader(testset, batch_size, shuffle=True)

    # Mostrar algunas características de los conjuntos de datos
    # Print out some stats about the training and test data
    print('Train data, number of images: ', len(trainset))
    print('Validation data, number of images: ', len(testset))

    # Nombrar las clases
    # specify the image classes
    classes = ['cactus', 'no_cactus']

    model = LeNet5(10)
    model.to(device)
    print(model)

    # define the criterion and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Cliclo principal de entrenamiento por épocas

    steps = 0
    running_loss = 0
    print_every = 40
    save_after = 100

    history_epoch = []
    history_train_loss = []
    history_validation_loss = []
    history_train_accuracy = []
    history_validation_accuracy = []

    import time
    tic = time.clock()

    for e in range(epochs):
        # Cambiamos a modo entrenamiento
        model.train()  
        
        for images, labels in trainloader:
            steps += 1
                
            optimizer.zero_grad()
            
            # wrap them in a torch Variable
            images, labels = Variable(images), Variable(labels)  
            
            images = images.to(device)
            labels = labels.to(device)
            
            output = model(images)
            loss = criterion(output, labels)
            # Backpropagation
            loss.backward()
            # Optimización
            optimizer.step()
            
            running_loss += loss.item()
        
        # Cambiamos a modo de evaluación
        model.eval()
        
        # Apagamos los gradientes, reduce memoria y cálculos
        with torch.no_grad():
            train_loss, train_accuracy = validation(model, trainloader, criterion, device)
            val_loss, val_accuracy = validation(model, testloader, criterion, device)
            
            train_loss, train_accuracy = train_loss, train_accuracy.cpu().numpy()
            val_loss, val_accuracy = val_loss, val_accuracy.cpu().numpy()
            
            train_accuracy = train_accuracy / len(trainloader)
            val_accuracy = val_accuracy / len(testloader)
            
            print("Epoch: {}/{}.. ".format(e+1, epochs),
                "Training Loss: {:.3f}.. ".format(train_loss),
                "Val. Loss: {:.3f}.. ".format(val_loss),
                "Train Accuracy: {:.3f}".format(train_accuracy),
                "Val. Accuracy: {:.3f}".format(val_accuracy))
        
        # guardamos parametros estadisticos
        history_epoch.append(e)
        history_train_loss.append(train_loss)
        history_validation_loss.append(val_loss)
        history_train_accuracy.append(train_accuracy)
        history_validation_accuracy.append(val_accuracy)
        
        # reseteamos la perdida actual
        running_loss = 0
        
        # gardamos parametros estadisticos
        if(e % save_after == 0):
            np.save('log/train_loss', history_train_loss)
            np.save('log/validation_loss', history_validation_loss)
            np.save('log/train_accuracy', history_train_accuracy)
            np.save('log/validation_accuracy', history_validation_accuracy)
            torch.save(model.state_dict(), 'log/weights.pth')
        
        # regresamos el grafo de la red a modo entrenamiento
        model.train()

    # calculamos el tiempo que tomo el entrenamiento
    toc = time.clock()
    print(toc - tic)


if __name__ == "__main__":
   argParser = argparse.ArgumentParser()
   argParser.add_argument("-f", "--folder", help="folder where dataset is")

   argParser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')

   args = argParser.parse_args()

   use_cuda = not args.no_cuda and torch.npu.is_available()
   device = torch.device("npu:0" if use_cuda else "cpu")

   folder = args.folder

   batch_size = 10

   main(folder, device, batch_size=batch_size)