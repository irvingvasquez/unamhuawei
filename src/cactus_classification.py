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


def main(folder, batch_size):
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
    print('Test data, number of images: ', len(testset))

    # Nombrar las clases
    # specify the image classes
    classes = ['cactus', 'no_cactus']

if __name__ == "__main__":
   argParser = argparse.ArgumentParser()
   argParser.add_argument("-f", "--folder", help="folder where dataset is")

   args = argParser.parse_args()

   folder = args.folder

   batch_size = 10

   main(folder, batch_size=batch_size)