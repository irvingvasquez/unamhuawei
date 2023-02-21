'''
Exploratory data analisys for cactus classification
@juan1rving
'''

import os
import argparse


def main(folder):
    folder_names = os.listdir(folder)
    n_classes = len(folder_names)
    print('Number of classes:', n_classes)
    for i, ff in enumerate(folder_names):
        print("class " + str(i) + ": " + folder_names[i])

if __name__ == "__main__":
   argParser = argparse.ArgumentParser()
   argParser.add_argument("-f", "--folder", help="folder where dataset is")

   args = argParser.parse_args()

   folder = args.folder

   main(folder)