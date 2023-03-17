from sklearn.ensemble import RandomForestClassifier
import os 
import pickletools
from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils.class_weight import compute_sample_weight

import datetime
import gzip
import _pickle as pickle
import re
import numpy as np
directory = "/home/mpc/LMILP/Datasets"
files = os.listdir(directory)
dates = []
N=20
dt=0.1
integer=int(0.1)
decimal=int(dt/0.1)
dt_str=str(integer)+str(decimal)
files = os.listdir(directory)
for file in files:
    #match = re.search("MILP_data_points_dt"+dt+"_N"+N+"_(.*).p", file)
    match = re.search("/home/mpc/LMILP/MILP_data_points_dt01_N20_(.*).p", file)
    if match:
        date_str = match.group(1)
        try:
            date = datetime.datetime.strptime(date_str,"%Y%m%d-%H%M%S")
            dates.append(date)

        except:
            a=1
dates.sort(reverse=True)
recent_date_str=dates[0].strftime("%Y%m%d-%H%M%S")

PIK="/home/mpc/LMILP/ProcessedFiles/data_set_pr_dt"+dt_str+"_N"+str(N)+"_"+recent_date_str+"_bin_N.p"
PIK1="/home/mpc/LMILP/ProcessedFiles/bin_map_pr_dt"+dt_str+"_N"+str(N)+"_"+recent_date_str+"_bin_N.p"
PIK2="/home/mpc/LMILP/ProcessedFiles/bin_labels_pr_dt"+dt_str+"_N"+str(N)+"_"+recent_date_str+"_bin_N.p"




with gzip.open(PIK, "rb") as f:
    p=pickle.Unpickler(f)
    dataset=p.load()
with gzip.open(PIK1, "rb") as f:
    p=pickle.Unpickler(f)
    bin_map_N=p.load()
with gzip.open(PIK2, "rb") as f:
    p=pickle.Unpickler(f)
    bin_labels_N=p.load()
rf_classifiers=[]
X_train=[dataset[i][0] for i in range(len(dataset))]
random_forest_model_paths=[]
for i in range(N):
    random_forest_model_paths.append("/home/mpc/LMILP/TrainedModels/rf_model_"+str(i)+"_dt"+dt_str+"_N"+str(N)+"_"+recent_date_str+".pkl")
for n in range(N):
    y_train=[dataset[i][2][n] for i in range(len(dataset))]
    # from sklearn.preprocessing import StandardScaler
    # scaler = StandardScaler()
    # X_scaled = scaler.fit_transform(X_train)
    # from sklearn.decomposition import PCA
    # pca = PCA(n_components=0.95)
    # X_pca = pca.fit_transform(X_scaled)balanced',np.unique(y_train
    class_weights = compute_class_weight(class_weight ="balanced", classes= np.unique(y_train), y= y_train)
    rf_classifier = RandomForestClassifier(n_estimators=10, class_weight=dict(enumerate(class_weights)),min_samples_leaf=2)
    rf_classifier.fit(X_train, y_train)
    train_accuracy = rf_classifier.score(X_train, y_train)
    print("Training Accuracy:", train_accuracy)
    with open(random_forest_model_paths[n], 'wb') as f:
        pickle.dump(rf_classifier, f)
a=1
