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
N=40
dt=0.1
# integer=int(0.1)
# decimal=int(dt/0.1)
# dt_str=str(integer)+str(decimal)
# files = os.listdir(directory)
# for file in files:
#     #match = re.search("MILP_data_points_dt"+dt+"_N"+N+"_(.*).p", file)
#     match = re.search("MILP_data_points_dt01_N20_(.*).p", file)
#     if match:
#         date_str = match.group(1)
#         try:
#             date = datetime.datetime.strptime(date_str,"%Y%m%d-%H%M%S")
#             dates.append(date)

#         except:
#             a=1
# dates.sort(reverse=True)
# recent_date_str=dates[0].strftime("%Y%m%d-%H%M%S")
dt_str="01"
recent_date_str="20230301-144802"
PIK="/home/mpc/LMILP/ProcessedFiles/data_set_pr_dt"+dt_str+"_N"+str(N)+"_"+recent_date_str+"_binaug.p"
PIK1="/home/mpc/LMILP/ProcessedFiles/bin_map_pr_dt"+dt_str+"_N"+str(N)+"_"+recent_date_str+"_binaug.p"
PIK2="/home/mpc/LMILP/ProcessedFiles/bin_labels_pr_dt"+dt_str+"_N"+str(N)+"_"+recent_date_str+"_binaug.p"




with gzip.open(PIK, "rb") as f:
    p=pickle.Unpickler(f)
    dataset=p.load()
with gzip.open(PIK1, "rb") as f:
    p=pickle.Unpickler(f)
    bin_map=p.load()
with gzip.open(PIK2, "rb") as f:
    p=pickle.Unpickler(f)
    bin_labels=p.load()
dataset_correct=[]
for k in bin_labels:
    dataset_k=[dataset[i] for i in bin_labels[k]]
    dataset_correct.extend(dataset_k)
#del bin_labels[482]
max_sp_num=max([len(x) for x in bin_map.values()])
X_train=[dataset_correct[i][0] for i in range(len(dataset_correct))]
y_train=[dataset_correct[i][2] for i in range(len(dataset_correct))]
# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X_train)
# from sklearn.decomposition import PCA
# pca = PCA(n_components=0.95)
# X_pca = pca.fit_transform(X_scaled)balanced',np.unique(y_train
class_weights = compute_class_weight(class_weight ="balanced", classes= np.unique(y_train), y= y_train)
rf_classifier = RandomForestClassifier(n_estimators=5, class_weight=dict(zip(np.unique(y_train), class_weights)), min_samples_leaf=2)
rf_classifier.fit(X_train, y_train)
train_accuracy = rf_classifier.score(X_train, y_train)
print("Training Accuracy:", train_accuracy)
random_forest_model_path="/home/mpc/LMILP/TrainedModels/rf_modeldt"+dt_str+"_N"+str(N)+"_"+recent_date_str+"_bin2.pkl"
with open(random_forest_model_path, 'wb') as f:
    pickle.dump(rf_classifier, f)
a=1
