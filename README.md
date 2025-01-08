# Predizione della Struttura Secondaria delle Proteine

Questo progetto implementa diversi modelli di deep learning per la predizione della struttura secondaria delle proteine (Protein Secondary Structure Prediction - PSSP), basandosi sul lavoro di [Zhou et al. (2014)](https://arxiv.org/abs/1403.1347).
L'obiettivo è confrontare l'efficacia di diverse architetture neurali, adattando le reti FCN ([Long et al. 2015](https://arxiv.org/abs/1411.4038)) e U-Net ([Ronneberger et al. 2015](https://arxiv.org/abs/1505.04597)) al dominio specifico della predizione della struttura delle proteine.

## Panoramica del Progetto

Il progetto implementa tre diverse architetture neurali per affrontare il problema della PSSP:
1. FCN 1D
2. U-Net 1D base
3. U-Net 1D ottimizzata

## Dataset e Preprocessing

Il progetto utilizza due dataset principali descritti da Zhou et al. (2014):
- **CullPDB 6133 filtered**: dataset di training contenente 6133 proteine
- **CB513**: dataset di test contenente 513 proteine

E' possibile scaricare i due dataset dal sito della [Uniwersytet Warszawski](https://lbs.cent.uw.edu.pl/pipred).

### Caratteristiche dei dati
Ogni proteina nel dataset include:
- One-hot encoding degli amminoacidi (20 features)
- Valori PSSM (Position-Specific Scoring Matrix) (20 features)
- Encoding posizionale (1 feature)
- Labels per 8 classi di struttura secondaria (Q8)

### Preprocessing
- Normalizzazione dei valori PSSM per proteina
- Aggiunta di encoding posizionale normalizzato
- Creazione di maschere per gestire sequenze di lunghezza variabile
- Trasposizione dei dati per ottimizzare le operazioni convolutive

## Modelli Implementati

### 1. FCN 1D
Implementazione monodimensionale della FCN descritta da Long et al. (2015):
- Encoder in stile VGG16 adattato a 1D
- Conversione dei layer fully connected in layer convoluzionali
- Predizione multi-scala con fusione di feature maps
- Batch Normalization e ReLU dopo ogni layer convoluzionale

### 2. U-Net 1D Base
Adattamento monodimensionale della U-Net originale:
- Architettura simmetrica encoder-decoder
- Skip connections tra encoder e decoder
- Batch Normalization e ReLU dopo ogni layer convolutivo
- Operazioni di max pooling e up-sampling
- Concatenazione delle feature maps attraverso le skip connections

### 3. U-Net 1D Ottimizzata
Versione migliorata della U-Net con ottimizzazioni specifiche per PSSP. Contiene tutte le caratteristiche della U-Net 1D base, con l'aggiunta di:
- Batch Normalization iniziale sull'input
- Dropout in ogni layer convolutivo
- Inizializzazione dei pesi He Kaiming
- Connessioni residuali nei blocchi convolutivi

## Training

### Configurazione
- Loss Function: Cross Entropy Loss con mascheramento
- Ottimizzatore: AdamW con weight decay
- Learning Rate: Scheduling adattivo con ReduceLROnPlateau
- Early Stopping: Monitoraggio della loss sul dataset CB513
- Batch Size: Configurabile via file YAML
- Gradient Clipping: Norm massima 1.0

### Metriche
- Loss sul training set
- Accuratezza Q8 (8 classi)
- Accuratezza Q3 (3 classi, opzionale)
- Monitoraggio con TensorBoard

## Configurazione

Ogni modello ha un file YAML di configurazione associato che permette di definire:
- Parametri dell'architettura
- Parametri di training
- Directory per i checkpoint
- Frequenza di logging e salvataggio
- Posizione dei dataset

## Requisiti
E' possibile installare i requisiti con:
```bash
pip3 install -r requirements.txt
```

## Utilizzo

```bash
# Training di un modello
python3 train.py models/model_config.yaml

# Visualizzazione dei risultati
tensorboard --logdir tensorboard/model/
```

## Struttura del Progetto
```
.
├── checkpoints/
├── data/
│   ├── cullpdb+profile_6133.npy.gz
│   └── cb513+profile_split1.npy.gz
├── models/
│   ├── m_fcn.py
│   ├── m_unet.py
│   ├── m_unet_optimized.py
│   ├── m_fcn.yaml
│   ├── m_unet.yaml
│   └── m_unet_optimized.yaml
├── tensorboard/
│   ├── fcn/
│   ├── unet/
│   └── unet_optimized/
├── dataset.py
└── train.py
```
