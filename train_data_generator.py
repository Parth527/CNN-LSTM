from data_loader import Dataloader
dataloader = Dataloader()
dataset = dataloader.ImportFile(path=r'C:\Users\pmodi\Downloads\training\training')
X_train, y_train = dataloader.prepare_input_data(dataset)
