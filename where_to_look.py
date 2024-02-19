import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("DEVICE is", DEVICE)

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" #!!! muss noch geändert werden

from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from model_network import Network, NetworkMultiLoss
import json
import cv2
from PIL import Image
from tqdm import tqdm
import random
import gdown

#Bild wird resized und in einen Tensor konvertiert
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# Zeigt die Aktivierungen des letzten conv layers
def PredictSingle(Imagepath, mode):
    image = Image.open(Imagepath)
    image = transform(image).unsqueeze(0)  # Dimension hinzufügen
    
    image = image.to('cuda')
    model = Network().to('cuda')
    
    # Modell laden
    if mode == 'Gender':
        model.load_state_dict(torch.load('Where-to-look_models\\model_gender_last.pt')['net_state_dict'])
    elif mode == "Malaria":
        model.load_state_dict(torch.load('Where-to-look_models\\model_malaria_last.pt')['net_state_dict'])
    else:
        raise ValueError('Mode not valid, please use "Malaria" or "Gender"')
    
    # Deaktiviert Funktionen, wie dropout
    model.eval()
    
    # Pred ist der Vektor der Vorhersage z.B.: [-0.0685,  0.1490]
    # Last_conv_features sind die featuremaps (256 14x14 maps)
    pred, last_conv_features = model(image, return_cam=True)
    
    # Sie Gewichtungen des fully connected layers (2 x 256)
    fc_weights = model.classifier[-1].weight.data
    
    # Vorhersage für das Bild
    _, predicted_class = torch.max(pred, 1)
    
    # Es werden nur die Gewichte für die richtige Klasse benötigt (1 x 256)
    # Mit den Gewichten der falschen Klasse könnte man gucken, was z.B. an einem Mann weiblich ist
    class_weights = fc_weights[predicted_class]
    
    # Eine 14x14 Matrix aus Nullen wird erstellt, welche danach befüllt wird
    cam = torch.zeros((last_conv_features.shape[2], last_conv_features.shape[3]), dtype=torch.float32).to('cuda')
    
    # Die Nullen werden durch die weights der richtigen prediction * last_conv_features für jede featuremap ersetzt
    # So entstehen 256 gewichtete featuremaps, welche alle zu einer 14x14 Matrix zusammenaddiert werden
    for i, w in enumerate(class_weights[0]):
        cam += w * last_conv_features[0, i]
    
    # Normalisieren (Alle Werte von 0 bis 1)
    cam = cam - torch.min(cam)
    cam = cam / torch.max(cam)
 
    cam = cam.to('cpu').detach().numpy() # Numpy array
    
    pred_class = predicted_class.item() # Von tensor([1], device='cuda:0') zu 1
    
    # Matrix wird hochskaliert, um sie über das Ausgangsbild zu legen
    cam_resized = cv2.resize(cam, (128, 128))
    
    # Eingabe Bild
    image_np = image.cpu().squeeze().numpy().transpose(1, 2, 0)
    
    # Bild und Heatmap plotten
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(image_np)
    plt.title('Original Image')
    
    plt.subplot(1, 2, 2)
    plt.imshow(image_np)
    plt.imshow(cam_resized, cmap='jet', alpha=0.5)
    plt.title(f'CAM Heatmap Overlay; Class: {pred_class}')
    
    plt.show()
    
# Plottet die Aktivierung in jedem Conv Block
def PredictMulti(Imagepath, end_or_every = 'End'):
    
    # Bild laden und verarbeiten
    image = Image.open(Imagepath)
    image = transform(image).unsqueeze(0).to('cuda')
    
    model = NetworkMultiLoss().to('cuda')
    
    # Auswählen, ob nur auf den loss der letzten Ebene oder den loss jeder Ebene optimiert wurde
    if end_or_every == 'End':
        model.load_state_dict(torch.load('Where-to-look_models\\model_malaria_every_single_loss.pt')['net_state_dict'])
    elif end_or_every == 'Every':
        model.load_state_dict(torch.load('Where-to-look_models\\model_malaria_every_multi_loss.pt')['net_state_dict'])
    else:
        raise ValueError('End or Every')
    
    model.eval()
    
    # Feature maps von allen convolutional layers (4 Tensoren)
    feature_maps = model(image, return_features=True) 
    
    # Vorhersage der letzten Ebene; Der Vorhersagen der Zwischenebenen werden nur beim Training benötigt
    with torch.no_grad():
        pred_vec = model(image, return_features=False).to('cpu')
    
    # Vorhergesagte Klassse ermitteln
    pred_probs = torch.softmax(pred_vec, dim=1)
    pred = torch.argmax(pred_probs, dim=1)

    pred_probs.cpu().numpy()
    pred = pred.cpu().numpy()

    if pred == 0:
        Pred = 'Uninfected'
    else:
        Pred = 'Infected'
        
    if  end_or_every == 'End': #!!! gucken ob klappt
        loss_text = 'Last layer loss'
    else:
        loss_text = 'Every layer loss'
    
    # 5 Subplots erstellen
    fig, axs = plt.subplots(1, len(feature_maps) + 1, figsize=(15, 5))
    fig.suptitle(f'Prediction: {Pred} \n{loss_text}', fontsize=16)

    axs[0].imshow(image.cpu().squeeze().numpy().transpose(1, 2, 0))
    axs[0].set_title('Original Image')
    
    # Für jede der 4 Tensoren mit featuremaps aus feature_maps wird der Durchschnitt aller featuremaps berechnet und geplottet
    for i, feature_map in enumerate(feature_maps):
        
        # Die Dimension der featuremaps sind ([1, 32, 64, 64]), ([1, 64, 32, 32]), ([1, 128, 16, 16]) und ([1, 256, 8, 8])
        # Nun bilden wir für die Dimension an Stelle 1 den Mittelwert, also bleiben (64, 64), (32, 32), (16, 16) und (8,8)
        selected_map = feature_map.mean(1).squeeze().cpu().detach().numpy()  # Average across channels
        
        # Diese Matrizen müssen wir normieren und auf die Größe unseres Bildes resizen
        selected_map = (selected_map - np.min(selected_map)) / (np.max(selected_map) - np.min(selected_map))
        resized_map = cv2.resize(selected_map, (128, 128))
        
        # Plotten
        axs[i + 1].imshow(image.cpu().squeeze().numpy().transpose(1, 2, 0))
        axs[i + 1].imshow(resized_map, cmap='jet', alpha=0.5)
        axs[i + 1].set_title(f'Layer {i+1} Activation')
    
    plt.show()

# Legt 100 Heatmaps eines Geschlechts übereinander
def PredictGender100Heatmaps(Imagepath):
    Heatmaps = []    

    image = Image.open(Imagepath)
    image = transform(image).unsqueeze(0)  # Add batch dimension
    
    image = image.to('cuda')
    
    model = Network().to('cuda')
    model.load_state_dict(torch.load('Where-to-look_models\\model_gender_last.pt')['net_state_dict']) #!!! gucken, ob der name stimmt

    model.eval()
    
    # Pred ist der Vektor der Vorhersage z.B.: [-0.0685,  0.1490]
    # Last_conv_features sind die featuremaps (256 14x14 maps)
    pred, last_conv_features = model(image, return_cam=True)
    
    # Sie Gewichtungen des fully connected layers (2 x 256)
    fc_weights = model.classifier[-1].weight.data
    
    # Vorhersage für das Bild
    _, predicted_class = torch.max(pred, 1)
    
    # Es werden nur die Gewichte für die richtige Klasse benötigt (1 x 256)
    # Mit den Gewichten der falschen Klasse könnte man gucken, was z.B. an einem Mann weiblich ist
    class_weights = fc_weights[predicted_class]
    
    # Die Nullen werden durch die weights der richtigen prediction * last_conv_features für jede featuremap ersetzt
    # So entstehen 256 gewichtete featuremaps, welche alle zu einer 14x14 Matrix zusammenaddiert werden
    cam = torch.zeros((last_conv_features.shape[2], last_conv_features.shape[3]), dtype=torch.float32).to('cuda')
    for i, w in enumerate(class_weights[0]):
        cam += w * last_conv_features[0, i]
    
    # Normalisieren (Alle Werte von 0 bis 1)
    cam = cam - torch.min(cam)
    cam = cam / torch.max(cam)
 
    cam = cam.to('cpu').detach().numpy()

    # Matrix wird hochskaliert, um sie über das Ausgangsbild zu legen
    cam_resized = cv2.resize(cam, (128, 128))
    
    # Matrizen werden in Heatmaps gespeichert
    Heatmaps.append(cam_resized)

    return np.squeeze(Heatmaps)

def PredictGender100():
    
    Gender_dataset_path = input('Path to Gender Dataset (...\\Dataset) \n-->')
    Dataset_Test_Male_Path = os.path.join(Gender_dataset_path, 'Test\\Male')
    Dataset_Test_Female_Path = os.path.join(Gender_dataset_path, 'Test\\Female')

    heatmappaths_male = []
    
    # Liste wird mit Pfaden aller Test-Bilder von Männern gefüllt
    for file in tqdm(os.listdir(Dataset_Test_Male_Path)):
        heatmappaths_male.append(os.path.join(Dataset_Test_Male_Path, file))
    
    # 100 Pfade auswählen
    random.shuffle(heatmappaths_male)
    heatmappaths_male = heatmappaths_male[:100]
    
    Heatmapsoutput = []
    
    # Heatmaps für alle Bilder erstellen
    for i in tqdm(heatmappaths_male):
        Heatmapsoutput.append(PredictGender100Heatmaps(i))
    
    # Array zum zusammen rechnen der Heatmaps
    accumulated_heatmap = np.zeros_like(Heatmapsoutput[0])
    
    # Alle Heatmaps werden addiert und der Mittelwert wird gebildet.
    for heatmap in tqdm(Heatmapsoutput):
        accumulated_heatmap += heatmap
        
    average_heatmap_male = accumulated_heatmap / len(Heatmapsoutput)
    
    # Plotten
    plt.imshow(average_heatmap_male, cmap='binary')
    plt.axis('off')
    plt.title("Male")
    plt.show()
    
    heatmappaths_female = []
    
    # Liste wird mit Pfaden aller Test-Bilder von Frauen gefüllt
    for file in tqdm(os.listdir(Dataset_Test_Female_Path)):
        heatmappaths_female.append(os.path.join(Dataset_Test_Female_Path, file))
    
    # 100 Pfade auswählen
    random.shuffle(heatmappaths_female)
    heatmappaths_female = heatmappaths_female[:100]
    
    # List to store the output heatmaps
    Heatmapsoutput = []
    
    # Heatmaps für alle Bilder erstellen
    for i in tqdm(heatmappaths_female):
        Heatmapsoutput.append(PredictGender100Heatmaps(i))
    
    # Array zum zusammen rechnen der Heatmaps
    accumulated_heatmap = np.zeros_like(Heatmapsoutput[0])
    
    # Alle Heatmaps werden addiert und der Mittelwert wird gebildet.
    for heatmap in tqdm(Heatmapsoutput):
        accumulated_heatmap += heatmap

    average_heatmap_female = accumulated_heatmap / len(Heatmapsoutput)
    
    # Plotten
    plt.imshow(average_heatmap_female, cmap='binary')
    plt.axis('off')
    plt.title("Female")
    plt.show()

# Legt zwei gespeicherte Bilder übereinander
def OverlayMaleFemale(Path_to_Male100, Path_to_Female100):
    import matplotlib.image as mpimg
    img1 = mpimg.imread(Path_to_Male100)
    img2 = mpimg.imread(Path_to_Female100)
    
    if img1.shape == img2.shape and img1.dtype == img2.dtype:
        # Mittelwert der Bilder bilden
        blended_img = (img1.astype('float') + img2.astype('float')) /2
    
        # Plotten
        plt.imshow(blended_img.astype(img1.dtype))
        plt.axis('off')
        plt.show()
    else:
        print("Images are not of the same size or type and cannot be blended directly.")

# Gewünschte Heatmaps werden berechnet
def Main():
    
    # Wenn keine modelle gefunden werden, können diese automatisch aus Google Drive heruntergeladen werden
    if not os.path.exists('Where-to-look_models'):
        user_input_download = ''
        while user_input_download != 'Y' and user_input_download != 'N':
            user_input_download = input('Do you want to download pretrained models from google drive? Please answer Y for yes or N for no \n-->').upper()
        if user_input_download == 'Y':
            url = "https://drive.google.com/drive/folders/1WUPvRI-5E139TkvZ3a44O8c85U7jZQPW?usp=sharing"
            gdown.download_folder(url)
        else:
            raise ValueError('No model found. Please use model_main.py to train a model or check if your models are in Where-to-look/Where-to-look_models/')
            
    # Benutzereingaben, um zu entscheiden was ausgeführt wird
    user_input_Dataset = ''
    while user_input_Dataset != 'A' and user_input_Dataset != 'B' and user_input_Dataset != 'C':
        user_input_Dataset = input('Gender or Malaria? Please answer A for gender, B for malaria or C if you want to use other features \n-->').upper()

    user_input_demo_or_own = ''
    while user_input_demo_or_own != 'A' and user_input_demo_or_own != 'B':
        user_input_demo_or_own = input('Do you want to use your own Picture or the demo pictures? Please answer A for a own picture or B for the demo pictures \n-->').upper()

    if user_input_Dataset == 'A': #Wenn der Gender Datensatz gewählt wurde
        if user_input_demo_or_own == 'A': #Wenn der Nutzer ein eigenes Bild verwenden möchte
            path = input('Enter the path of your file \n-->')
            PredictSingle(path, mode = 'Gender')
        else: #Wenn der Nutzer die Demo Bilder verwenden möchte
            # je Geschlecht werden 10 random Bilder aus dem test Ordner ausgewählt
            test_filenames_male = []
            
            Gender_dataset_path = input('Path to Gender Dataset (...\\Dataset) \n-->')
            print(Gender_dataset_path)
            Gender_dataset_path_male = os.path.join(Gender_dataset_path, 'Test\\Male')
            Gender_dataset_path_female = os.path.join(Gender_dataset_path, 'Test\\Female')
            
            for file in tqdm(os.listdir(Gender_dataset_path_male)): 
               test_filenames_male.append(os.path.join(Gender_dataset_path_male, file))

            random.shuffle(test_filenames_male)

            test_filenames_male = test_filenames_male[:10]

            for i in test_filenames_male:
                PredictSingle(i,mode='Gender')
                
                
            test_filenames_female = []

            for file in tqdm(os.listdir(Gender_dataset_path_female)):
               test_filenames_female.append(os.path.join(Gender_dataset_path_female, file))

            random.shuffle(test_filenames_female)

            test_filenames_female = test_filenames_female[:10]

            for i in test_filenames_female:
                PredictSingle(i,mode='Gender')
    
    elif user_input_Dataset == 'B': #Wenn der Malaria Datensatz gewählt wurde
        
    #Für die Filenames aus der jeweiligen json Datei
        user_input_layer_activision = ''
        while user_input_layer_activision != 'A' and user_input_layer_activision != 'B':
            user_input_layer_activision = input('Do you want to see the activision of the last Conv layer or the activisions of every layer? Please answer A for the last layer only or B for every layer\n-->').upper()
        
        if user_input_layer_activision == 'A': #Wenn der Nutzer nur die Heatmap des letzten Conv layers sehen möchte
            if user_input_demo_or_own == 'A': #Wenn der Nutzer ein eigenes Bild verwenden möchte
                path = input('Enter the path of your file \n-->')
                PredictSingle(path, mode = 'Malaria')
            else: #Wenn der Nutzer die Demo Bilder verwenden möchte
                # Filenames der nicht zum Training verwendeten Bilder werden aus der JSON Datei ausgelesen
                Malaria_dataset_path = input('Path to Malaria Dataset (...\\Malaria_Cell_Images) \n-->')
                with open('Data\\malaria_testing_filenames_last.json', 'r') as file:    #!!! filenames ändern von single/ multi zu last/every
                    loaded_data = json.load(file)

                InfectedFilenames = list(loaded_data['InfectedFilenames'])
                UninfectedFilenames = list(loaded_data['UninfectedFilenames'])
                
                for file in InfectedFilenames:
                    PredictSingle(os.path.join(Malaria_dataset_path,"cell_images\\Parasitized",file), mode = "Malaria")
                for file in UninfectedFilenames:
                    PredictSingle(os.path.join(Malaria_dataset_path, "cell_images\\Uninfected",file), mode = "Malaria")
                    
        else: #Wenn der Nutzer die Heatmaps aller Conv layer sehen möchte
            user_input_loss_last_or_every = ''
            while user_input_loss_last_or_every != 'A' and user_input_loss_last_or_every != 'B':
                user_input_loss_last_or_every = input('Do you want to use the model with the loss calculated only for the last layer or for each layer? Please answer A for the last layer or B for every layer \n-->').upper()
            
            if user_input_loss_last_or_every == 'A': # Wenn der Nutzer das Modell auswählt, welches auf den Loss des letzten Conv Blocks trainiert wurde
                if user_input_demo_or_own == 'A': #Wenn der Nutzer ein eigenes Bild verwenden möchte
                    path = input('Enter the path of your file \n-->')
                    PredictMulti(path, end_or_every = 'End')
                else: #Wenn der Nutzer die Demo Bilder verwenden möchte
                    # Filenames der nicht zum Training verwendeten Bilder werden aus der JSON Datei ausgelesen
                    Malaria_dataset_path = input('Path to Malaria Dataset (...\\Malaria_Cell_Images) \n-->')
                    with open('Data\\malaria_testing_filenames_every.json', 'r') as file: #!!! filenames chekcen
                        loaded_data = json.load(file)

                    InfectedFilenames = list(loaded_data['InfectedFilenames'])
                    UninfectedFilenames = list(loaded_data['UninfectedFilenames'])

                    for file in InfectedFilenames:
                        PredictMulti(os.path.join(Malaria_dataset_path,"cell_images\\Parasitized",file), end_or_every = 'End')

                    for file in UninfectedFilenames:
                        PredictMulti(os.path.join(Malaria_dataset_path,"cell_images\\Uninfected",file), end_or_every = 'End')
                        
            else: # Wenn der Nutzer das Modell auswählt, welches den Loss in jeder Ebene minimiert
                if user_input_demo_or_own == 'A': #Wenn der Nutzer ein eigenes Bild verwenden möchte
                    path = input('Enter the path of your file \n-->')
                    PredictMulti(path, end_or_every = 'Every')
                else: #Wenn der Nutzer die Demo Bilder verwenden möchte
                    # Filenames der nicht zum Training verwendeten Bilder werden aus der JSON Datei ausgelesen
                    Malaria_dataset_path = input('Path to Malaria Dataset (...\\Malaria_Cell_Images) \n-->')
                    with open('Data\\malaria_testing_filenames_every_every.json', 'r') as file:  #!!! filenames chekcen
                        loaded_data = json.load(file)

                    InfectedFilenames = list(loaded_data['InfectedFilenames'])
                    UninfectedFilenames = list(loaded_data['UninfectedFilenames'])

                    for file in InfectedFilenames:
                        PredictMulti(os.path.join(Malaria_dataset_path,"cell_images\\Parasitized",file), end_or_every = 'Every')
                        
                    for file in UninfectedFilenames:
                        PredictMulti(os.path.join(Malaria_dataset_path, "cell_images\\Uninfected",file), end_or_every = 'Every')
                        
    else: #Wenn der Nutzer die sonstigen Funktionen nutzen möchte
        # Zusatzfunktionen
        user_input_Gender100_OverlayMaleFemale = ''
        while user_input_Gender100_OverlayMaleFemale != 'A' and user_input_Gender100_OverlayMaleFemale != 'B':
            user_input_Gender100_OverlayMaleFemale = input('Do you want to use the Gender100 function or the OverlayMaleFemale function? Please answer A for Gender100 or B for OverlayMaleFemale \n-->').upper()
        if user_input_Gender100_OverlayMaleFemale == 'A':
            PredictGender100()
            
        else:
            if user_input_demo_or_own == 'A':
                OverlayMaleFemale(input('Path to Male100 plot \n-->'), input('Path to Female100 plot \n-->'))
            else:
                OverlayMaleFemale('Data\Figure_Gender100_male.png','Data\Figure_Gender100_female.png') #!!! Path zu bildern auf Github
                
if __name__ == "__main__":
    Main()