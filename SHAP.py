#!/omics/odcf/analysis/OE0167_projects/dachs_genetic_data_platform/Jupyter/tf/bin/python3.9

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

import numpy as np
import pandas as pd
import gc
import random as rn
import os

import tensorflow as tf
from tensorflow import keras
#from keras import regularizers
from tensorflow.keras.models import Sequential, load_model, Model
#from tensorflow.keras.layers import Dense, Dropout, ActivityRegularization, Input
#from tensorflow.keras.regularizers import l2
#from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import shap

#self-defined function
#exec(open("/omics/odcf/analysis/OE0167_projects/dachs_genetic_data_platform/Jupyter/Deepsurv/trial_code_current/deepsurv_and_ML/tutorials/xiao_functionjf.py").read())

# set random seed
SEED = 1234
np.random.seed(SEED)
tf.random.set_seed(SEED)
rn.seed(SEED)
os.environ['PYTHONHASHSEED'] = '0'

# load model
model = load_model('/omics/odcf/analysis/OE0167_projects/dachs_genetic_data_platform/Jupyter/tuneresult_10fold/individual/model/H1cv1', compile = False)

# import necessary data
train_x = pd.read_csv("/omics/odcf/analysis/OE0167_projects/dachs_genetic_data_platform/Jupyter/data_10fold/individualCpGs/traincv1_x_alldeath.csv")
test_x = pd.read_csv("/omics/odcf/analysis/OE0167_projects/dachs_genetic_data_platform/Jupyter/data_10fold/individualCpGs/testcv1_x_alldeath.csv")

train_cpgs = train_x.iloc[:, 4:]
test_cpgs = test_x.iloc[:, 4:]

# Standardization 
X_scaler = StandardScaler().fit(train_cpgs)
X_train = X_scaler.transform(train_cpgs)
X_test = X_scaler.transform(test_cpgs)

## explore different shap values
# 1. prepare background datasets
# 1.1 1000 random (same)training data
bg_1000t = pd.read_csv("/omics/odcf/analysis/OE0167_projects/dachs_genetic_data_platform/Jupyter/multimodal_data/lite_20kcpgs/t1000_bg.csv")
bg_1000t = X_scaler.transform(bg_1000t)

# 1.2 all zeros
bg_zero = np.zeros((1, train_cpgs.shape[1]))

# 1.3 normal tissue
bg_nm = pd.read_csv("/omics/odcf/analysis/OE0167_projects/dachs_genetic_data_platform/Jupyter/multimodal_data/lite_20kcpgs/nm_bg.csv")
bg_nm = X_scaler.transform(bg_nm)

### 3. deep shap
## 3.1 1000 training
shap.explainers._deep.deep_tf.op_handlers["AddV2"] = shap.explainers._deep.deep_tf.passthrough

explainer = shap.DeepExplainer(model, bg_1000t)

shap_values = explainer.shap_values(X_test)

shaps_cur=np.mean(np.abs(shap_values[0]), axis=0)

shap_df = pd.DataFrame({'Feature_name': train_cpgs.columns,
                        'Shapv_deepbt1000_cv1': shaps_cur})

shap_df.to_csv('/omics/odcf/analysis/OE0167_projects/dachs_genetic_data_platform/Jupyter/tuneresult_10fold/individual/shap/h1cv1_dsb1000r.csv',
        index=False)

del explainer
del shap_values
del shaps_cur
del shap_df

gc.collect()

## 3.2 all zeros
explainer = shap.DeepExplainer(model, bg_zero)

shap_values = explainer.shap_values(X_test)

shaps_cur=np.mean(np.abs(shap_values[0]), axis=0)
shap_df = pd.DataFrame({'Feature_name': train_cpgs.columns,
                        'Shapv_deepbzero_cv1': shaps_cur})

shap_df.to_csv('/omics/odcf/analysis/OE0167_projects/dachs_genetic_data_platform/Jupyter/tuneresult_10fold/individual/shap/h1cv1_dsbzero.csv',
        index=False)

del explainer
del shap_values
del shaps_cur
del shap_df

gc.collect()

## 3.3 normal tissue
explainer = shap.DeepExplainer(model, bg_nm)

shap_values = explainer.shap_values(X_test)

shaps_cur=np.mean(np.abs(shap_values[0]), axis=0)

shap_df = pd.DataFrame({'Feature_name': train_cpgs.columns,
                        'Shapv_deepbnm_cv1': shaps_cur})

shap_df.to_csv('/omics/odcf/analysis/OE0167_projects/dachs_genetic_data_platform/Jupyter/tuneresult_10fold/individual/shap/h1cv1_dsbnm.csv',
        index=False)

del explainer
del shap_values
del shaps_cur
del shap_df

gc.collect()

# 4 integrated Gradient
# 4.1 1000 training
explainer = shap.GradientExplainer(model, [bg_1000t])

shap_values = explainer.shap_values(X_test)

shaps_cur=np.mean(np.abs(shap_values[0]), axis=0)

shap_df = pd.DataFrame({'Feature_name': train_cpgs.columns,
                        'Shapv_gradientbt1000_cv1': shaps_cur})

shap_df.to_csv('/omics/odcf/analysis/OE0167_projects/dachs_genetic_data_platform/Jupyter/tuneresult_10fold/individual/shap/h1cv1_igb1000r.csv',
        index=False)
        

del explainer
del shap_values
del shaps_cur
del shap_df
gc.collect()

# 4.2 all zero
explainer = shap.GradientExplainer(model, [bg_zero])
shap_values = explainer.shap_values(X_test)
shaps_cur=np.mean(np.abs(shap_values[0]), axis=0)

shap_df = pd.DataFrame({'Feature_name': train_cpgs.columns,
                        'Shapv_gradientbzero_cv1': shaps_cur})

shap_df.to_csv('/omics/odcf/analysis/OE0167_projects/dachs_genetic_data_platform/Jupyter/tuneresult_10fold/individual/shap/h1cv1_igbzero.csv', index=False)

del explainer
del shap_values
del shaps_cur
del shap_df

gc.collect()

# 4.3 normal tissue
explainer = shap.GradientExplainer(model, [bg_nm])

shap_values = explainer.shap_values(X_test)

shaps_cur=np.mean(np.abs(shap_values[0]), axis=0)

shap_df = pd.DataFrame({'Feature_name': train_cpgs.columns,
                        'Shapv_gradientbnm_cv1': shaps_cur})

shap_df.to_csv('/omics/odcf/analysis/OE0167_projects/dachs_genetic_data_platform/Jupyter/tuneresult_10fold/individual/shap/h1cv1_igbnm.csv', index=False)

del explainer
del shap_values
del shaps_cur
del shap_df

gc.collect()






