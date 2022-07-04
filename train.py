from data_loader import Dataloader
from Evaluation import *
from CNN_LSTM_model import get_1DCNN, Input_preprocess
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.utils import to_categorical
import numpy as np
skf = StratifiedKFold(n_splits=10, random_state=12, shuffle=True)

dataloader = Dataloader()
dataset = dataloader.ImportFile(path=r'C:\Users\pmodi\Downloads\training\training')
X_train, y_train = dataloader.prepare_input_data(dataset)
# i=0
#
# for train_index,test_index in skf.split(X_train, y_train):
#     X_tr, X_te = X_train[train_index], X_train[test_index]
#     y_tr, y_te = y_train[train_index], y_train[test_index]
#     X_tr = np.expand_dims(X_tr, axis=-1)
#     X_te = np.expand_dims(X_te, axis=-1)
#     y_tr = to_categorical(y_tr)
#     y_true = y_te
#     y_te = to_categorical(y_te)
#     i=i+1
#     print("Training Fold ",i)

X_train, X_test, y_train, y_test,y_true = Input_preprocess(X_train=X_train,y_train=y_train)
history, model = get_1DCNN(x_train=X_train, y_train=y_train,x_test=X_test,y_test=y_test, epochs=150)
get_summary(x_test=X_test, y_true=y_true, model=model)
Plot_Acc_and_Loss(history=history,x_test=X_test,y_test=y_test,model=model)

