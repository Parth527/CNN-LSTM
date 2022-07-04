# -*- coding: utf-8 -*-
"""

Skript testet das vortrainierte Modell


@author: Christoph Hoog Antink, Maurice Rohr
"""

import csv
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model
import os
from typing import List, Tuple
from data_loader import Dataloader

###Signatur der Methode (Parameter und Anzahl return-Werte) darf nicht verändert werden
def predict_labels(ecg_leads : List[np.ndarray], fs : float, ecg_names : List[str], model_name : str='model.npy',is_binary_classifier : bool=False) -> List[Tuple[str,str]]:
    '''
    Parameters
    ----------
    model_name : str
        Dateiname des Models. In Code-Pfad
    ecg_leads : list of numpy-Arrays
        EKG-Signale.
    fs : float
        Sampling-Frequenz der Signale.
    ecg_names : list of str
        eindeutige Bezeichnung für jedes EKG-Signal.
    model_name : str
        Name des Models, kann verwendet werden um korrektes Model aus Ordner zu laden
    is_binary_classifier : bool
        Falls getrennte Modelle für F1 und Multi-Score trainiert werden, wird hier übergeben, 
        welches benutzt werden soll
    Returns
    -------
    predictions : list of tuples
        ecg_name und eure Diagnose
    '''

# Euer Code ab hier  
    model = load_model('1DCNN_best_model_94F1score_to_submit.h5')
    predictions = list()
    dataloader = Dataloader()
    for ecg_name, ecg in zip(ecg_names, ecg_leads):

        ECG_signal = dataloader.WindowSelection(signal = ecg, win_size=9000, StartPoint=0)
        y_pred_prob = model.predict(np.expand_dims(np.expand_dims(ECG_signal,axis=-1),axis=0))
        y_pred = y_pred_prob.argmax(axis=1)

        if y_pred == 1:
            predictions.append((ecg_name, 'A'))
        else:
            predictions.append((ecg_name, 'N'))
#------------------------------------------------------------------------------    
    return predictions # Liste von Tupels im Format (ecg_name,label) - Muss unverändert bleiben!
                               
                               
        
