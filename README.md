# Where-to-look
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


![image](https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=blue)

## Einleitung
Where to look ist ein Python Projekt, welches im Rahmen des DAISY Studiums entwickelt wurde. Es dient dazu mittels Class Activision Mapping (CAM) jene Regionen in einem Bild sichtbar zu machen, die eine hohe Aktivierung bei der Klassifizierung durch ein Convolutional Neural Network (CNNs), aufweisen. So entsteht eine Heatmap, durch welche ein Objekt in einem Bild lokalisiert werden kann.

## Funktionen
- **Modelltraining**: Die benötigten Modelle können alle lokal trainiert werden
- **Objekterkennung**: Identifizierung von Malaria-infizierten Zellen und Unterscheidung zwischen männlichen und weiblichen Bildern.
- **Objektlokalisierung**: Visualisierung der spezifischen Bildbereiche, die zur Klassifizierung beitragen, durch eine Heatmap.

## Daten
Das Programm benötigt mindestens einen der beiden Datensätze. Für volle Funktionalität werden beide benötigt:

**Malaria Datensatz:**

https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria?resource=download

**Gender Datensatz:**

https://www.kaggle.com/datasets/ashishjangra27/gender-recognition-200k-images-celeba?resource=download

##Installation

```bash
git clone https://github.com/Julimaeder/Where-to-look.git
cd Where-to-look
pip install -r requirements.txt
```

Zum trainieren der Modelle anschließend 
```bash
python model_main.py
```
in der Konsole oder einer beliebigen IDE ausführen. Alternativ können die trainierten Modelle automatisch in der where_to_look.py Datei von Google Drive heruntergeladen werden. Dazu einfach den Anweisungen folgen.

Zum erstellen der Heatmaps (und ggf. Dwonload der Modelle) anschließend
```bash
python where_to_look.py
```
in der Konsole oder einer IDE ausführen und jeweils den Anweisungen folgen. Eine IDE, welche plots gut darstellen und verwalten kann bietet sich hierzu an.

