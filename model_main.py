import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("DEVICE is", DEVICE)

import os
# Bei Problemen mit dem Enviroment ggf. ausführen: os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

if __name__ == "__main__":
    
    from torch.utils.data import DataLoader
    from model_data import GenderSet, MalariaSet, DatasetCache
    from model_network import Network, NetworkMultiLoss
    from model_trainer import CheckpointTrainer
    
    # User inputs um zu entscheiden welches Modell trainiert werden soll
    Input_Gender_or_Malaria = ''
    while Input_Gender_or_Malaria != 'G' and Input_Gender_or_Malaria != 'M':
        Input_Gender_or_Malaria = input("Do you want to train the Network with the Gender or Malaria Dataset? Please input G or M\n-->").upper()
        
    # Wenn noch nicht vorhanden wird ein Ordner für die Modelle erstellt
    if not os.path.exists('Where-to-look_models'):
        os.makedirs('Where-to-look_models')

    loss = torch.nn.CrossEntropyLoss()
    
    if Input_Gender_or_Malaria == 'G': #Gender
    
        net = Network().to(DEVICE)
        LastvsEvery = 'last'
        
        # Bilder laden
        dataset = DatasetCache(
            GenderSet(max_samples_per_class=16384))
        
        dataset_val = DatasetCache(
            GenderSet(max_samples_per_class=2048, mode = 'Val'))
        
        dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
        dataloader_val = DataLoader(dataset_val, batch_size=128, shuffle=True)
        
        # Trainieren
        trainer = CheckpointTrainer(net, loss, f"Where-to-look_models\\model_gender_{LastvsEvery}.pt")
        trainer.train(dataloader, dataloader_val)
        
    else: # Malaria
        Input_last_or_every_layer_Network = ''
        while Input_last_or_every_layer_Network != 'L' and Input_last_or_every_layer_Network != 'E':
            Input_last_or_every_layer_Network = input("Do you want to see the activision of the last Conv layer or the activisions of every layer? Please input L for the last layer only or E every layer\n-->").upper()
        
        if Input_last_or_every_layer_Network == 'L':
            net = Network().to(DEVICE)
            LastvsEvery = 'last'
        else:
            net = NetworkMultiLoss().to(DEVICE)
            LastvsEvery = 'every'
            
        #Bilder laden
        dataset = DatasetCache(
          MalariaSet(max_samples_per_class=8192,single_or_multi = LastvsEvery))
        
        dataset_val = DatasetCache(
          MalariaSet(max_samples_per_class=2048, mode = 'Val',single_or_multi =  LastvsEvery))

        dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
        dataloader_val = DataLoader(dataset_val, batch_size=128, shuffle=True)
        
        #Je nach input verschiedene Modelle trainieren
        if Input_last_or_every_layer_Network == 'L':
            trainer = CheckpointTrainer(net, loss, chkpt_path = f"Where-to-look_models\\model_malaria_{LastvsEvery}.pt")
        else:
            Input_loss_at_end_or_every_layer = ''
            while Input_loss_at_end_or_every_layer != 'L' and Input_loss_at_end_or_every_layer != 'E':
                Input_loss_at_end_or_every_layer = input("Should the loss be calculated only for the last layer or for each layer? Please input L for last or E for each\n-->").upper()
                
            if Input_loss_at_end_or_every_layer == 'L':
                trainer = CheckpointTrainer(net, loss, chkpt_path = f"Where-to-look_models\\model_malaria_{LastvsEvery}_single_loss.pt")
                
            elif Input_loss_at_end_or_every_layer == 'E':
                # MalariaSet every_layer_loss = True mit übergeben
                dataset = DatasetCache(
                  MalariaSet(max_samples_per_class=8192,single_or_multi = LastvsEvery, every_layer_loss = True))
                dataset_val = DatasetCache(
                  MalariaSet(max_samples_per_class=2048, mode = 'Val',single_or_multi =  LastvsEvery, every_layer_loss = True))
                
                trainer = CheckpointTrainer(net, loss, f"Where-to-look_models\\model_malaria_{LastvsEvery}_multi_loss.pt",Loss_jede_ebene = True)
            
        trainer.train(dataloader, dataloader_val)
