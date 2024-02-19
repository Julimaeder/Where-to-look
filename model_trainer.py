import torch
from torch.utils.data import DataLoader
from model_data import GenderSet, DatasetCache
from model_network import Network
from tqdm import tqdm

# Trainiert das Modell
class Trainer:
    def __init__(self, network, loss_function, Loss_jede_ebene = False):
        self.network = network
        self.loss_function = loss_function
        
        self.init_optimizer()
        self.init_scheduler()
        
        self.Loss_jede_ebene = Loss_jede_ebene
    
    def init_optimizer(self):
        self.optim = torch.optim.Adam(self.network.parameters(), lr=0.0001)
    
    def init_scheduler(self):
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optim, 0.95)
    
    def epoch(self, dataloader, training, epoch=0):
        # tqdm Leiste zur besseren Übersicht
        bar = tqdm(dataloader)
        
        # Training oder evaluating
        if training:
            self.network.train()
            name="train"
        else:
            self.network.eval()
            name="val"
    
        total_loss = 0
        correct = 0
        cnt = 0
    
        # Über die gesamte Epoche iterieren
        for batch, labels in bar:
        
            if training:
                #setzt die Gradienten der Modellparameter auf Null, ansonsten addieren sie sich bei mehreren durchläufen
                self.optim.zero_grad()
      
            #Loss wird für jede Ebene berechnet, nicht nur für die Letzte
            if self.Loss_jede_ebene == True:
                #Features & outputs aller Ebenen
                features = self.network(batch, return_outputs=True)
      
                accumulated_loss = 0
                
                #processed_output sind die outputs aller Ebenen
                for _, processed_output in features:
                    # Labels müssen Eindimensional sein
                    if labels.dim() > 1:
                        labels = labels.squeeze()
                    if labels.dim() > 1:
                        raise ValueError("Labels tensor is not 1D after squeezing.")
      
                    # Loss für die Vorhersage jeder Ebene wird berechnet und addiert
                    layer_loss = self.loss_function(processed_output, labels)

                    if accumulated_loss == 0:
                        accumulated_loss = layer_loss
                    else:
                        accumulated_loss += layer_loss
                        
                #Ausgabe der letzten ebene wird gespeichert
                res = processed_output
                
                total_loss += accumulated_loss.item()
              
            else:
                res = self.network(batch) #Prediction
                labels = labels.reshape(-1)
                loss = self.loss_function(res, labels)
                total_loss += loss.item()
    
            # Anzahl der korrekten Vorhersagen zur Berechnung der accuracy
            correct += torch.sum(torch.argmax(res, dim=1) == labels).item()
        
            # Gesamtzahl der durch gelaufenen batches
            cnt += batch.shape[0]
        
            bar.set_description(f"ep: {epoch:.0f} ({name}), loss: {1000.0*total_loss / cnt:.3f}, acc: {100.0*correct/cnt:.2f}%")
        
            # Backpropagation beim Training 
            if training:
                if self.Loss_jede_ebene:
                    accumulated_loss.backward() # Backpropagation wird auf den loss aller Ebenen angewandt
                else:
                    loss.backward() #Backpropagation für den einzelnen loss
                self.optim.step() # Netzwerkparameter aktualisieren
        
        return 1000.0 * total_loss / cnt, 100.0*correct/cnt

class CheckpointTrainer(Trainer):
    def __init__(self, network, loss_function, chkpt_path, epochs = 100 ,Loss_jede_ebene = False):
      super().__init__(network, loss_function, Loss_jede_ebene)

      self.ep = 0
      self.chkpt_path = chkpt_path
      self.best_val_acc = 0
      self.epochs = epochs
      try:
          chkpt = torch.load(self.chkpt_path) #Checkpoint laden, wenn vorhanden
          self.network.load_state_dict(chkpt["net_state_dict"])
          self.optim.load_state_dict(chkpt["optim_state_dict"])
          self.scheduler.load_state_dict(chkpt["scheduler_state_dict"])
          self.best_val_acc = chkpt["best_val_acc"]
          self.ep = chkpt["epoch"]
      except:
          print("Could not find checkpoint, starting from scratch")

    def train(self, loader_train, loader_val):
      while self.ep <= self.epochs:
        train_loss, train_acc = self.epoch(loader_train, True, self.ep) # Trainer Klasse wird aufgerufen
        if loader_val is not None:
          val_loss, val_acc = self.epoch(loader_val, False, self.ep) #Training wird auf False gesetzt
        else:
          val_acc = train_loss, train_acc
        self.scheduler.step()
        
        self.ep += 1

        self.best_val_acc = val_acc
        #Checkpoint erstellen
        torch.save({
            "net_state_dict": self.network.state_dict(),
            "optim_state_dict": self.optim.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_acc": self.best_val_acc,
            "epoch": self.ep
        }, self.chkpt_path)

if __name__ == "__main__":
    from model_main import DEVICE
    dataset = DatasetCache(
      GenderSet(max_samples_per_class=4096))
    
    dataset_val = DatasetCache(
       GenderSet(max_samples_per_class=1024, mode = 'Val'))
    
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    dataloader_val = DataLoader(dataset_val, batch_size=32, shuffle=True)
    
    net = Network().to(DEVICE)
    loss = torch.nn.CrossEntropyLoss()

    trainer = CheckpointTrainer(net, loss, "model.pt")
    trainer.train(dataloader, dataloader_val)
        