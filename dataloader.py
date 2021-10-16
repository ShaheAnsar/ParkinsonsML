import numpy as np
import nibabel as nib
from nibabel.processing import conform
import pandas as pd
import sys
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.optimizers import SGD
from keras.models import Sequential
from keras.layers.core import Dense
import cv2

class DataLoader:
    def __init__(self, dataset, dataset_root, preprocessor=None):
        if dataset == "neurocon":
            self.data = self.load_neurocon(dataset_root, preprocessor)


    def load_neurocon(self, dataset_root, preprocessor):
        info_file = "neurocon_patients.tsv"
        info_data = pd.read_csv(os.path.join(dataset_root, info_file), sep="\t")
        prefix = "sub-"
        # Contains the directories for each patient/control
        dirs = [prefix + code for code in info_data["code"]]
        # Get the anat MRI filenames
        anat_postfix = "_T1w.nii.gz"
        anat_dir = "anat"
        anat_names = [prefix + code + anat_postfix
                      for code in info_data["code"]]
        anat_paths = [os.path.join(dataset_root, dirs[i],
                                   anat_dir, anat_names[i])
                      for i in range(len(anat_names))]
        # Get the fMRI filenames
        func_dir = "func"
        func_run_postfix = lambda i: f"_task-resting_run-{i}_bold.nii.gz"
        func_r1_names = [prefix + code + func_run_postfix(1)
                         for code in info_data["code"]]
        func_r2_names = [prefix + code + func_run_postfix(2)
                         for code in info_data["code"]]
        func_r1_paths = [os.path.join(dataset_root, dirs[i], func_dir,
                                      func_r1_names[i])
                         for i in range(len(func_r1_names))]
        #func_r2_paths = [os.path.join(dataset_root, dirs[i], func_dir,
        #                              func_r2_names[i])
        #                 for i in range(len(func_r2_names))]
        #func_paths = [func_r1_paths, func_r2_paths]
        func_paths = [func_r1_paths]

        # Load the data in
        anat_X = np.stack([preprocessor("anat", nib.load(fname))
                           for fname in anat_paths])
        func_X = np.stack([
            [preprocessor("func", nib.load(fname)) for fname in func_paths[0]],
            #[preprocessor("func", nib.load(fname)) for fname in func_paths[1]]
        ])
        anat_Y = np.array([1 if "patient" in name else 0
                           for name in anat_names])
        func_Y = np.array([
            [1 if "patient" in name else 0 for name in func_r1_names],
            #["patient" in name for name in func_r2_names]
        ])

        return ((anat_X, anat_Y), (func_X, func_Y))

    def load_taowu(self, dataset_root, preprocessing):
        pass

size = 50
def neurocon_pp(_type, nib_obj):
    global size
    
    if _type == "anat":
        nib_obj = conform(nib_obj, out_shape=(size, size, size))
        fdata = nib_obj.get_fdata()
        return fdata.flatten()
    else:
        return np.zeros(1)

dl = DataLoader("neurocon", "../datasets/neurocon", neurocon_pp)

((X,y), _)= dl.data
X /= np.max(X)
(trainX, testX, trainY, testY) = train_test_split(X, y)
print(f"trainY Shape: {trainY.shape}")
print("Model Creation...")
#sgd = SGD(0.001)

model = Sequential()
model.add(Dense(1024, input_shape=(size**3,), activation="relu"))
model.add(Dense(1024, activation="relu"))
model.add(Dense(512, activation="relu"))
model.add(Dense(64, activation="relu"))
model.add(Dense(1, activation="sigmoid"))
print("Compiling Model...")
sgd = SGD(0.001)
model.compile(loss="binary_crossentropy", optimizer=sgd,
              metrics=["accuracy"])
print("Fitting Model...")
H = model.fit(trainX, trainY, epochs=1000, validation_data=(testX, testY))
p = model.predict(testX)
p.reshape(( p.shape[0], ))
print("Predicted Values")
print(p)
p[p < 0.5] = 0
p[p > 0] = 1
print("Predicted Values after step")
print(p)
print(classification_report(testY, p, target_names=["NP", "PD"]))
print("Expected Values")
print(testY)
print("Predictions from trainX")
print(model.predict(trainX))

