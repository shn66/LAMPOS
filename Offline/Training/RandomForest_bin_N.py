import sys
sys.path.append("/home/mpc/LMILP/LAMPOS/")
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight
from utils import get_recentdate_str
import gzip
import _pickle as pickle
import numpy as np

# Select the horizon and sampling time

N=40
dt=0.1

# Pick the most recent processed files with the selected horizon and sampling time

dataset_path = "Offline/Datasets"
recent_date_str,dt_str=get_recentdate_str(dataset_path=dataset_path,N=N,dt=dt)

# Dataset and labels dictionary loading

PIK="Offline/ProcessedFiles/data_set_pr_dt"+dt_str+"_N"+str(N)+"_"+recent_date_str+"_bin_N.p"
PIK1="Offline/ProcessedFiles/bin_map_pr_dt"+dt_str+"_N"+str(N)+"_"+recent_date_str+"_bin_N.p"
PIK2="Offline/ProcessedFiles/bin_labels_pr_dt"+dt_str+"_N"+str(N)+"_"+recent_date_str+"_bin_N.p"
with gzip.open(PIK, "rb") as f:
    p=pickle.Unpickler(f)
    dataset=p.load()
with gzip.open(PIK1, "rb") as f:
    p=pickle.Unpickler(f)
    bin_map_N=p.load()
with gzip.open(PIK2, "rb") as f:
    p=pickle.Unpickler(f)
    bin_labels_N=p.load()


# Defining the paths in which load/save the trained model

random_forest_model_paths=[]
for i in range(N):
    random_forest_model_paths.append("Offline/TrainedModels/rf_model_"+str(i)+"_dt"+dt_str+"_N"+str(N)+"_"+recent_date_str+".pkl")

# Training RF models for each time-step

rf_classifiers=[]
X_train=[dataset[i][0] for i in range(len(dataset))]
for n in range(N):
    y_train=[dataset[i][2][n] for i in range(len(dataset))]     # Model definition
    class_weights = compute_class_weight(class_weight ="balanced", classes= np.unique(y_train), y= y_train)
    rf_classifier = RandomForestClassifier(n_estimators=10, class_weight=dict(enumerate(class_weights)),min_samples_leaf=2)
    rf_classifier.fit(X_train, y_train)                         # Model training
    train_accuracy = rf_classifier.score(X_train, y_train)
    print("Training Accuracy:", train_accuracy)
    with open(random_forest_model_paths[n], 'wb') as f:         # Saving trained model 
        pickle.dump(rf_classifier, f)

