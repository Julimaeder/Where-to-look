import torch
from torch.utils.data import DataLoader
from model_data import GenderSet
import numpy as np

# Zusammenfassung von Operationen
class Down(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super(Down, self).__init__()
        
        self.seq  = torch.nn.Sequential(
            torch.nn.Conv2d(in_features, out_features, kernel_size=(5,5), padding="same"),
            torch.nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)), #2x2 Felder werden immer betrachtet; Kernel geht immer 2 Schrittte weiter
            torch.nn.BatchNorm2d(num_features = out_features),
            torch.nn.ReLU()
            )

    def forward(self, x):
        return self.seq(x)

class Network(torch.nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        self.features = torch.nn.Sequential(
            Down(in_features=3, out_features=32),  # 32x64x64
            torch.nn.Dropout2d(0.2),
            Down(in_features=32, out_features=64),  # 64x32x32
            torch.nn.Dropout2d(0.2),
            Down(in_features=64, out_features=128),  # 128x16x16
            torch.nn.Dropout2d(0.2),
           
            #Außerhalb der Down Klasse, um auf den Conv2d Layer zugreifen zu können
            torch.nn.Conv2d(128, 256, kernel_size=(5,5), padding="same"),
            torch.nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)),
            torch.nn.BatchNorm2d(num_features = 256),
            torch.nn.ReLU(),  
            )

        self.gap = torch.nn.AdaptiveAvgPool2d((1, 1))  # Global Average Pooling
        self.classifier = torch.nn.Sequential(
            torch.nn.Flatten(),  # 1x1x256 zu 256
            torch.nn.Dropout(0.5),
            torch.nn.Linear(256, 2)  # Auf 2 Neuronen runter zur Klassifizierung
            )

    def forward(self, x, return_cam=False):
        features = self.features(x)
        pooled_features = self.gap(features)
        output = self.classifier(pooled_features)
        if return_cam:
            return output, features  #Features vom letzten Conv Layer
        else:
            return output

#Kann die Aktivierung jedes Conv layers zurück geben; kann loss jedes Conv layers minimieren
class NetworkMultiLoss(torch.nn.Module):
    def __init__(self):
        super(NetworkMultiLoss, self).__init__()

        self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=(5,5), padding="same")
        self.pool1 = torch.nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        self.bn1 = torch.nn.BatchNorm2d(num_features=32)
        self.relu1 = torch.nn.ReLU()
        self.gap1 = torch.nn.AdaptiveAvgPool2d((1, 1))
        
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=(5,5), padding="same")
        self.pool2 = torch.nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        self.bn2 = torch.nn.BatchNorm2d(num_features=64)
        self.relu2 = torch.nn.ReLU()
        self.gap2 = torch.nn.AdaptiveAvgPool2d((1, 1))
        
        self.conv3 = torch.nn.Conv2d(64, 128, kernel_size=(5,5), padding="same")
        self.pool3 = torch.nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        self.bn3 = torch.nn.BatchNorm2d(num_features=128)
        self.relu3 = torch.nn.ReLU()
        self.gap3 = torch.nn.AdaptiveAvgPool2d((1, 1))
        
        self.conv4 = torch.nn.Conv2d(128, 256, kernel_size=(5,5), padding="same")
        self.pool4 = torch.nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        self.bn4 = torch.nn.BatchNorm2d(num_features=256)
        self.relu4 = torch.nn.ReLU()
        self.gap4 = torch.nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(256, 2)
        )
        # 256 ist der erwartete Input für den classifier
        self.transform1 = torch.nn.Linear(32, 256)
        self.transform2 = torch.nn.Linear(64, 256)
        self.transform3 = torch.nn.Linear(128, 256)

    def forward(self, x, return_features=False, return_outputs=False):
        #Alle Opertionen ohne GAP, weil nur die Featuremaps gebraucht werden
        x1 = self.relu1(self.bn1(self.pool1(self.conv1(x))))
        x2 = self.relu2(self.bn2(self.pool2(self.conv2(x1))))
        x3 = self.relu3(self.bn3(self.pool3(self.conv3(x2))))
        x4 = self.relu4(self.bn4(self.pool4(self.conv4(x3))))

        if return_features:
            return x1, x2, x3, x4  # Featuremaps aller Layer
        
        if return_outputs:
            # GAP wird auf jeden output angewandt
            gap1 = self.gap4(x1)
            gap2 = self.gap4(x2)
            gap3 = self.gap4(x3)
            gap4 = self.gap4(x4)
        
            # Flatten, um von 1,1,256 auf 256 zu kommen
            flat1 = torch.flatten(gap1, 1)
            flat2 = torch.flatten(gap2, 1)
            flat3 = torch.flatten(gap3, 1)
            flat4 = torch.flatten(gap4, 1)
        
            # Blöcke 1-3 auf die richtige size bringen für den classifier
            t1 = self.transform1(flat1)
            t2 = self.transform2(flat2)
            t3 = self.transform3(flat3)
        
            # Klassifizieren (binär)
            out1 = self.classifier(t1)
            out2 = self.classifier(t2)
            out3 = self.classifier(t3)
            out4 = self.classifier(flat4)
           
            return (x1, out1), (x2, out2), (x3, out3), (x4, out4)

        x = self.gap4(x4)  
        x = self.classifier(x)
        return x

if __name__ == "__main__":
    from model_main import DEVICE
    dataset = GenderSet(max_samples_per_class=2000)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    net = Network().to(DEVICE)

    total_parameters = 0
    model_parameters = filter(lambda p: p.requires_grad, net.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f"Network has {params} total parameters")
    
    batch, labels = dataloader.__iter__().__next__()
    x = net(batch)
    print(x.shape)