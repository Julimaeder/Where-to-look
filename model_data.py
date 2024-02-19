import os
import torch
from torchvision.io import read_image
from torchvision import transforms
from torch.utils.data import Dataset,DataLoader
from tqdm import tqdm
import random
from torchvision.utils import make_grid
import numpy as np
import json
from model_main import DEVICE

#Bild wird resized und integer in float werte konvertiert
resize_transform = transforms.Compose([
    transforms.Resize((128, 128), antialias=True),transforms.ConvertImageDtype(torch.float32)
])

# Lädt den Malaria Datensatz und splittet in train und validation Daten
class MalariaSet(Dataset):
    FOLDERMALARIA = ''
    def __init__(self, max_samples_per_class = None, mode = 'Train', single_or_multi = '', every_layer_loss = False):
        self.data = []
        self.Infected_Filenames_test = []
        self.Uninfected_Filenames_test = []
        self.single_or_multi = single_or_multi
        self.every_layer_loss = every_layer_loss

        Infected = []
        Uninfected = []
        
        # Pfad zum Ordner
        if MalariaSet.FOLDERMALARIA == '':
            MalariaSet.FOLDERMALARIA = input('Path to Malaria Dataset (...\\Malaria_Cell_Images): ')
            if os.path.exists(MalariaSet.FOLDERMALARIA) and os.path.isdir(MalariaSet.FOLDERMALARIA):
                MalariaSet.FOLDERMALARIA = os.path.join(MalariaSet.FOLDERMALARIA, 'cell_images\\')
            else:
                raise ValueError('Could not find the Malaria Dataset')
        
        # Zwei Listen mit allen Filenames von infizierten und nicht infizierten Zellen
        for file in tqdm(os.listdir(os.path.join(MalariaSet.FOLDERMALARIA, "Parasitized"))):
            if file.endswith('.png'):
                Infected.append(os.path.join(MalariaSet.FOLDERMALARIA, "Parasitized", file))
        for file in tqdm(os.listdir(os.path.join(MalariaSet.FOLDERMALARIA, "Uninfected"))):
            if file.endswith('.png'):
                Uninfected.append(os.path.join(MalariaSet.FOLDERMALARIA, "Uninfected", file))
        
        # Daten werden in Train und Validation gesteilt und in ein numpy array konvertiert
        Infected_filenames = np.array(Infected)
        Uninfected_filenames = np.array(Uninfected)
        train_ratio = 0.8  # 80% training, 20% testing

        # Anzahl der Daten berechnen
        Infected_num_train_files = int(len(Infected_filenames) * train_ratio) 
        Uninfected_num_train_files = int(len(Uninfected_filenames) * train_ratio)

        np.random.shuffle(Infected_filenames)
        np.random.shuffle(Uninfected_filenames)
        
        Infected_train = Infected_filenames[:Infected_num_train_files]
        Infected_val = Infected_filenames[Infected_num_train_files:-10]
        self.Infected_Filenames_test = Infected_filenames[-10:] #validation
        
        Uninfected_train = Uninfected_filenames[:Uninfected_num_train_files]
        Uninfected_val = Uninfected_filenames[Uninfected_num_train_files:-10]
        self.Uninfected_Filenames_test = Uninfected_filenames[-10:] #validation
        
        # Nur die Filenames speichern
        self.Infected_Filenames_test = [os.path.basename(path) for path in self.Infected_Filenames_test]
        self.Uninfected_Filenames_test = [os.path.basename(path) for path in self.Uninfected_Filenames_test]
        
        # Test Filenames als json Datei speichern
        data_to_save = {
        'InfectedFilenames': self.Infected_Filenames_test,  # Convert to list if they are numpy arrays
        'UninfectedFilenames': self.Uninfected_Filenames_test
        }
        if self.every_layer_loss == True:
            filename = f'malaria_testing_filenames_{self.single_or_multi}_every.json'
        else:
            filename = f'malaria_testing_filenames_{self.single_or_multi}.json'
        with open(filename, 'w') as file:
            json.dump(data_to_save, file)
        print(f'Testing pictures saved to {filename}')
        
        if mode == 'Train':
            Infected = Infected_train
            Uninfected = Uninfected_train
        elif mode == 'Val':
            Infected = Infected_val
            Uninfected = Uninfected_val
        else:
            raise ValueError("Invalid mode")
        
        # Liste kürzen, wenn max_samples_per_class angegeben ist
        if max_samples_per_class is not None:

            random.shuffle(Infected)
            random.shuffle(Uninfected)
    
            Infected = Infected[:max_samples_per_class]
            Uninfected = Uninfected[:max_samples_per_class]
        
        # Daten mit Label als Tupel in data speichern --> [(filename,Label),(filename,Label),...]
        for path in Uninfected:
            self.data.append((path, 0))
        
        for path in Infected:
            self.data.append((path, 1))
            
    def __len__(self):
        return len(self.data)

    # Indexiert den Datensatz, verarbeitet das Bild und gibt es mit Label zurück
    def __getitem__(self, idx):
      # Path und Label des Datenpunktes
      path, label = self.data[idx]

      image = read_image(path).to(DEVICE)

      # Resize
      image = resize_transform(image)
      label = torch.Tensor([label]).type(torch.LongTensor).to(DEVICE).reshape(-1)

      return image, label

#Lädt den Gender Datensatz; Schon in train, test und validation geteilt
class GenderSet(Dataset):
    FOLDERGENDER = ''
    
    def __init__(self, max_samples_per_class = None, mode = 'Train'):
        self.data = []

        male = []
        female = []
        
        # Pfad zum Ordner
        if GenderSet.FOLDERGENDER == '':
            GenderSet.FOLDERGENDER = input('Path to Gender Dataset (...\\Dataset): ')
            if not os.path.exists(GenderSet.FOLDERGENDER) and os.path.isdir(GenderSet.FOLDERGENDER):
                raise ValueError('Could not find the Gender Dataset')
            
        TRAIN_SET_FOLDER = os.path.join(GenderSet.FOLDERGENDER, 'Train\\')
        VAL_SET_FOLDER = os.path.join(GenderSet.FOLDERGENDER, 'Validation\\')
        
        #Training und Validation sind schon in verschiedenen Ordnern
        if mode == 'Train':
            folder = TRAIN_SET_FOLDER
        elif mode == 'Val':
            folder = VAL_SET_FOLDER
        else:
            raise ValueError("Invalid mode")
        
        # Listen mit allen Paths/Filenames
        for file in tqdm(os.listdir(os.path.join(folder, "Female"))):
           female.append(os.path.join(folder, "Female", file))
        for file in tqdm(os.listdir(os.path.join(folder, "Male"))):
           male.append(os.path.join(folder, "Male", file))
        
        # Liste kürzen, wenn max_samples_per_class angegeben ist
        if max_samples_per_class is not None:
            
            random.shuffle(male)
            random.shuffle(female)
        
            male = male[:max_samples_per_class]
            female = female[:max_samples_per_class]
            
         # Daten mit Label als Tupel in data speichern --> [(filename,Label),(filename,Label),...]
        for path in female:
            self.data.append((path, 0))
        
        for path in male:
            self.data.append((path, 1))
            
    def __len__(self):
        return len(self.data)
    
    # Indexiert den Datensatz, verarbeitet das Bild und gibt es mit Label zurück
    def __getitem__(self, idx):

      path, label = self.data[idx]

      image = read_image(path).to(DEVICE)

      image = resize_transform(image)
      label = torch.Tensor([label]).type(torch.LongTensor).to(DEVICE).reshape(-1)

      return image, label
  
    # Daten werden in self.cache gespeichert, sobald sie einmal angefragt wurden und können von da aus direkt erneut 
    # abgerufen werden, anstatt sie neu zu laden
class DatasetCache(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.cache = [None for _ in range(len(self.dataset))] #Liste die len(dataset) lang ist mit None Werten

    def __len__(self):
        return len(self.dataset)
    
    # Datenpunkt an der Stelle index wird ausgegeben. Falls noch nicht vorhanden wird in self.cache an der stelle 
    # index der dazugehörige Datenpunkt aus dataset gespeichert
    def __getitem__(self, index):
        item = self.cache[index]
        if item is None:
            item = self.dataset[index]
            self.cache[index] = item

        return item

# Zeigt einen Batch an Bildern an
if __name__ == "__main__":

    from matplotlib import pyplot as plt

    dataset = MalariaSet(max_samples_per_class=20)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    batch, label = dataloader.__iter__().__next__()
    
    grid = make_grid(batch, 8, padding=4).permute(1,2,0)

    plt.imshow(grid.cpu())
    plt.show()