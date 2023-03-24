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

PIK="Offline/ProcessedFiles/data_set_pr_dt"+dt_str+"_N"+str(N)+"_"+recent_date_str+".p"
PIK1="Offline/ProcessedFiles/bin_map_pr_dt"+dt_str+"_N"+str(N)+"_"+recent_date_str+".p"
PIK2="Offline/ProcessedFiles/bin_labels_pr_dt"+dt_str+"_N"+str(N)+"_"+recent_date_str+".p"
with gzip.open(PIK, "rb") as f:
    p=pickle.Unpickler(f)
    dataset=p.load()
with gzip.open(PIK1, "rb") as f:
    p=pickle.Unpickler(f)
    bin_map=p.load()
with gzip.open(PIK2, "rb") as f:
    p=pickle.Unpickler(f)
    bin_labels=p.load()

# Model definition

max_sp_num=max([len(x) for x in bin_map.values()])
X_train=[dataset[i][0] for i in range(len(dataset))]
y_train=[dataset[i][2] for i in range(len(dataset))]
class_weights = compute_class_weight(class_weight ="balanced", classes= np.unique(y_train), y= y_train)
rf_classifier = RandomForestClassifier(n_estimators=5, class_weight=dict(zip(np.unique(y_train), class_weights)), min_samples_leaf=2)

# Traininig 

rf_classifier.fit(X_train, y_train)

# Evaluation

train_accuracy = rf_classifier.score(X_train, y_train)
print("Training Accuracy:", train_accuracy)

# Saving Trained model

random_forest_model_path="Offline/TrainedModels/rf_modeldt"+dt_str+"_N"+str(N)+"_"+recent_date_str+"_bin.pkl"
with open(random_forest_model_path, 'wb') as f:
    pickle.dump(rf_classifier, f)
a=1
