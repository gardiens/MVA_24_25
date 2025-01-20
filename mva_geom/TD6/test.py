path_train_data = "/raid/home/bournez_pie/mva_geom/mva_geom_24/TD6/db_train.raw"
path_test_data = "/raid/home/bournez_pie/mva_geom/mva_geom_24/TD6/db_test.raw"
path_train_label = "/raid/home/bournez_pie/mva_geom/mva_geom_24/TD6/label_train.txt"  # load the train data , those are images of size 56x56x3
import numpy as np

X_data = np.fromfile(path_train_data, dtype=np.int16)
Y_train = np.fromfile(path_train_data, dtype=np.int16)
# load the train data , those are images of size 56x56x3
import numpy as np
import torch

X_data = np.fromfile(path_train_data, dtype=np.int16)
Y_train = np.fromfile(path_train_data, dtype=np.int16)
import psutil

print(f"Memory Usage: {psutil.Process().memory_info().rss / (1024 ** 2):.2f} MB")
X_data = np.fromfile(path_train_data, dtype="uint8")
X_data = X_data.reshape(-1, 56, 56, 3)
print("yes")
X_data = X_data / 255
# X_test = torch.tensor(np.fromfile(path_test_data, dtype='uint8'))/255
# X_test = X_test.reshape(-1, 56, 56, 3)
# X_test.shape

# #load the txt
# Y_train= torch.tensor(np.loadtxt(path_train_label, dtype=np.int16))
# print(Y_train)
