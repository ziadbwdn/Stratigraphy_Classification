# Library Importing (NOTE: PLEASE INSTALL SCIKIT-BIO FOR COMPOSITIONAL DATA ANALYSIS PURPOSES)

import numpy as np # for linear algebra operation
import pandas as pd # for data processing purposes
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from skbio.stats.composition import ilr, closure

# For modelling
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier # 1
from sklearn.tree import DecisionTreeClassifier # 2
from sklearn.ensemble import BaggingClassifier # 2
from sklearn.linear_model import LogisticRegression # 2
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn import svm
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelBinarizer
from itertools import cycle
from sklearn.metrics import RocCurveDisplay

# import from local library
import read_data
import index_to_drop
import drop_below_20
import split_input_output
import split_train_test
import preprocess_data
import accuracy_mx_report
import EDA_Box_Plot

data = read_data(fname = 'DataSet_Thaba_Classification.csv')
# checking the data 
data.head()

# declaring index to drop
idx_to_drop = index_to_drop(X = data)

# Check the dropped data index at first (JUST RUN THE CODE)
print(f'Number of index to drop:', len(idx_to_drop))
idx_to_drop

# dropping the index
data_dropped = data.drop(index=idx_to_drop)
data_dropped = data_dropped.reset_index(drop=True)

# Check the dropped data index at first (JUST RUN THE CODE)
print (f'Number of index before get dropped:', data.shape, '\n', f'And number of index after get dropped:', data_dropped.shape)

# rename MG4_ to MG4
data_dropped['Stratigraphy'] = data_dropped['Stratigraphy'].replace('MG4 ', 'MG4')

# continue drop necessary data
data_dropped_1 = drop_below_20(X=data_dropped, target = 'Stratigraphy')

# EDA Brief step

EDA_Box_Plot(X = data_dropped_1, 
             columns = [col for col in data_dropped_1 if col.endswith('_%')] + [col for col in data_dropped_1 if col.endswith('_ppm')]
            )

# Based on the distribution of P_% values on scatter plot above, we can consider this one is also an anomaly
P_ano_idx = data[data['P_%'] > 1].index

# dropping P_ano_idx
data_dropped_2 = data_dropped_1.drop(index=P_ano_idx)
data_dropped_2 = data_dropped_2.reset_index(drop=True)
data_dropped_3 = data_dropped_2.drop(['Au_ICP_ppm'], axis=1)
data_dropped_3 = data_dropped_3.reset_index(drop=True)

X, y = split_input_output(data=data_dropped_3, target_col='Stratigraphy')

# Split the data
# First, split the train & not train
X_train, X_not_train, y_train, y_not_train = split_train_test(X=X, y=y, test_size = 0.2, stratify = y, seed=123)

# Then, split the valid & test
X_valid, X_test, y_valid, y_test = split_train_test(X=X_not_train, y=y_not_train, test_size = 0.5, stratify = y_not_train, seed = 123)

# Validation step
print(len(X_train)/len(X))  # should be 0.8
print(len(X_valid)/len(X))  # should be 0.1
print(len(X_test)/len(X))   # should be 0.1

# preprocessing data step

X_train_cleaned, cat_imputer, cat_encoder, percent_closure, percent_ilr = preprocess_data(X_train, 
                                                                                          dataset_type='train')
X_valid_cleaned, _, _, _, _ = preprocess_data(X_valid, 
                                              cat_imputer=cat_imputer, 
                                              cat_encoder=cat_encoder, 
                                              dataset_type='valid')

X_test_cleaned, _, _, _, _ = preprocess_data(X_test, 
                                             cat_imputer=cat_imputer,  
                                             cat_encoder=cat_encoder, 
                                             dataset_type='test')


# Modelling step - Baseline model

# Make Dummy Classification for baseline model
dummy_clf = DummyClassifier(strategy = "stratified")

# Lakukan fit, untuk data y_train saja
dummy_clf.fit(X = X_train,
              y = y_train)

y_pred_dummy = dummy_clf.predict(X_train)

accuracy_mx_report(y_true = y_train, y_pred = y_pred_dummy, model = dummy_clf)


## Random Forest Method

#use encoded y_true data
from sklearn.preprocessing import LabelEncoder

# Initialize LabelEncoder
le = LabelEncoder()

# Apply label encoding to the target variable
y_train_encoded = le.fit_transform(y_train)
y_valid_encoded = le.transform(y_valid)
y_test_encoded = le.transform(y_test)

model_rf1 = RandomForestClassifier(random_state=42, class_weight='balanced')
model_rf1.fit(X_train_cleaned, y_train_encoded)

y_pred_train_rf = model_rf1.predict(X_test_cleaned)
y_pred_valid_rf = model_rf1.predict(X_test_cleaned)
y_pred_test_rf = model_rf1.predict(X_test_cleaned)

accuracy_mx_report(y_true = y_train_encoded, y_pred = y_pred_train_rf, model = model_rf1)
accuracy_mx_report(y_true = y_valid_encoded, y_pred = y_pred_valid_rf, model = model_rf1)
accuracy_mx_report(y_true = y_test_encoded, y_pred = y_pred_test_rf, model = model_rf1)

## Decision Tree with Bagging

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier

# Bagging - Decision Tree Model
B = [10, 20, 50, 70, 100, 120, 150]
params = {"n_estimators": B}

# Buat gridsearch
dt = DecisionTreeClassifier()
bagging_tree = BaggingClassifier(estimator = dt,
                                random_state = 123)

bagging_cv = GridSearchCV(estimator = bagging_tree,
                          param_grid = params,
                          cv = 5,
                          scoring = "accuracy")

# Fit grid search cv
bagging_cv.fit(X_train_cleaned, y_train_encoded)

# Best params
bagging_cv.best_params_

# Refit the bagging tree
bagging_tree = BaggingClassifier(n_estimators = bagging_cv.best_params_["n_estimators"],
                                random_state = 123)

model_bgt = bagging_tree.fit(X_train_cleaned, y_train_encoded)
model_bgt

# make y_pred

y_pred_bgd_train = model_bgt.predict(X_train_cleaned)
y_pred_bgd_valid = model_bgt.predict(X_valid_cleaned)
y_pred_bgd_test = model_bgt.predict(X_test_cleaned)

# make accuracy report
accuracy_dt_train = accuracy_mx_report(y_true = y_train_encoded, y_pred = y_pred_bgd_train, model = model_bgt)
accuracy_dt_valid = accuracy_mx_report(y_true = y_valid_encoded, y_pred = y_pred_bgd_valid, model = model_bgt)
accuracy_dt_test = accuracy_mx_report(y_true = y_test_encoded, y_pred = y_pred_bgd_test, model = model_bgt)

## Multinomial Logistic Regression

mlr = LogisticRegression(random_state=0, 
                         multi_class='multinomial', 
                         class_weight = dict(class_weight), 
                         solver='newton-cg')

model_mlr = mlr.fit(X_train_cleaned, y_train)

y_train_pred_mlr = model_mlr.predict(X_train_cleaned)
y_valid_pred_mlr = model_mlr.predict(X_valid_cleaned)
y_test_pred_mlr = model_mlr.predict(X_test_cleaned)

# print the tunable parameters (They were not tuned in this example, everything kept as default)
params = model_mlr.get_params()
print(params)

# Experimentation a bit with GridSearchCV with fitting the data

search_params = {"penalty": ['l2', None],
                 "C": [0.2,0.4,0,6,0.8,1,2,3]}

mlr_cv = GridSearchCV(estimator = model_mlr,
                      param_grid = search_params,
                      cv = 5)

# Data Fitting
mlr_cv.fit(X = X_train_scaled,
           y = y_train)

# Best params
mlr_cv.best_params_

# Create best model
mlr_best = LogisticRegression(multi_class='multinomial', 
                                penalty = mlr_cv.best_params_["penalty"],
                                C = mlr_cv.best_params_["C"],
                                solver = 'newton-cg',
                                random_state = 0)

model_mlr = mlr_best.fit(X_train_cleaned, y_train)

# output y_pred model multinomial logreg

y_train_pred_mlr = model_mlr.predict(X_train_cleaned)
y_valid_pred_mlr = model_mlr.predict(X_valid_cleaned)
y_test_pred_mlr = model_mlr.predict(X_test_cleaned)

params = model_mlr.get_params()
print(params)

# declare accuracy score, confusion matrix, and report for each dataset
accuracy_mx_report(y_true = y_train, y_pred = y_train_pred_mlr, model = model_mlr)
accuracy_mx_report(y_true = y_valid, y_pred = y_valid_pred_mlr, model = model_mlr)
accuracy_mx_report(y_true = y_test, y_pred = y_test_pred_mlr, model = model_mlr)

# Print model parameters (save it for later)
print('Intercept: \n', model_mlr.intercept_)
print('Coefficients: \n', model_mlr.coef_)

# array for coefficients (save it for later)
np.exp(model_mlr.coef_)


# LDA Classification

# using default solver (svd)
lda_model = LinearDiscriminantAnalysis (n_components = 7)

# dengan menggunakan solver lsqr
lda_model2 = LinearDiscriminantAnalysis (n_components = 7, 
                                         solver='lsqr', 
                                         shrinkage=None).fit(X_train_cleaned, y_train)

# making y_pred on model 1
y_pred_lda_train = lda_model.predict(X_train_cleaned)
y_pred_lda_valid = lda_model.predict(X_valid_cleaned)
y_pred_lda_test = lda_model.predict(X_test_cleaned)

# making y_pred on model 2
y_pred_lda2_train = lda_model2.predict(X_train_cleaned)
y_pred_lda2_valid = lda_model2.predict(X_valid_cleaned)
y_pred_lda2_test = lda_model2.predict(X_test_cleaned)

accuracy_mx_report(y_true = y_train, y_pred = y_pred_lda1_valid, model = model_lda1)
accuracy_mx_report(y_true = y_train, y_pred = y_pred_lda2_valid, model = model_lda2)

accuracy_mx_report(y_true = y_valid, y_pred = y_pred_lda1_valid, model = model_lda1)
accuracy_mx_report(y_true = y_valid, y_pred = y_pred_lda2_valid, model = model_lda2)

accuracy_mx_report(y_true = y_test, y_pred = y_pred_lda1_test, model = model_lda2)
accuracy_mx_report(y_true = y_test, y_pred = y_pred_lda2_test, model = model_lda2)

# Declare y_pred_proba on train data
y_pred_trainproba_rf = model_rf1.predict_proba(X_train_cleaned)
y_pred_trainproba_bgt = model_bgt.predict_proba(X_train_cleaned)
y_pred_trainproba_mlr = model_mlr.predict_proba(X_train_cleaned)
y_pred_trainproba_lda = lda_model2.predict_proba(X_train_cleaned)

# Declare y_pred_proba on valid data
y_pred_validproba_rf = model_rf1.predict_proba(X_valid_cleaned)
y_pred_validproba_bgt = model_bgt.predict_proba(X_valid_cleaned)
y_pred_validproba_mlr = model_mlr.predict_proba(X_valid_cleaned)
y_pred_validproba_lda = lda_model2.predict_proba(X_valid_cleaned)

# Declare y_pred_proba on test data
y_pred_testproba_rf = model_rf1.predict_proba(X_test_cleaned)
y_pred_testproba_bgt = model_bgt.predict_proba(X_test_cleaned)
y_pred_testproba_mlr = model_mlr.predict_proba(X_test_cleaned)
y_pred_testproba_lda = lda_model2.predict_proba(X_test_cleaned)

# Calculate log loss train
log_loss_trainrf = log_loss(y_train_encoded, y_pred_trainproba_rf)
log_loss_trainbgt = log_loss(y_train_encoded, y_pred_trainproba_bgt)
log_loss_trainmlr = log_loss(y_train, y_pred_trainproba_mlr)
log_loss_trainlda = log_loss(y_train, y_pred_trainproba_lda)

# Calculate log loss valid
log_loss_validrf = log_loss(y_valid_encoded, y_pred_validproba_rf)
log_loss_validbgt = log_loss(y_valid_encoded, y_pred_validproba_bgt)
log_loss_validmlr = log_loss(y_valid, y_pred_validproba_mlr)
log_loss_validlda = log_loss(y_valid, y_pred_validproba_lda)

# Calculate log loss test
log_loss_testrf = log_loss(y_test_encoded, y_pred_testproba_rf)
log_loss_testbgt = log_loss(y_test_encoded, y_pred_testproba_bgt)
log_loss_testmlr = log_loss(y_test, y_pred_testproba_mlr)
log_loss_testlda = log_loss(y_test, y_pred_testproba_lda)

# Report Dataframe
accuracy_train = [accuracy_train_dummy, accuracy_rf_train, accuracy_dt_train, accuracy_mlr_train, accuracy_lda_train]
accuracy_valid = [accuracy_valid_dummy, accuracy_rf_valid, accuracy_dt_valid, accuracy_mlr_valid, accuracy_lda_valid]
accuracy_test = [accuracy_test_dummy, accuracy_rf_test, accuracy_dt_test, accuracy_mlr_test, accuracy_lda_test]
log_loss_train = [0.0, log_loss_trainrf, log_loss_trainbgt, log_loss_trainmlr, log_loss_trainlda]
log_loss_valid = [0.0, log_loss_validrf, log_loss_validbgt, log_loss_validmlr, log_loss_validlda]
log_loss_test = [0.0, log_loss_testrf, log_loss_testbgt, log_loss_testmlr, log_loss_testlda]
indexes = ["dummy classifier","random forest", "bagged decision tree","multinomial logistic regression", "linear discriminant"]

summary_df = pd.DataFrame({"Accuracy Train Report":accuracy_train,
                           "Accuracy Valid Report":accuracy_valid,
                           "Accuracy Test Report": accuracy_test, 
                           "Log Loss Train Report": log_loss_train,
                           "Log Loss Valid Report": log_loss_valid,
                           "Log Loss Test Report": log_loss_test},
                          index = indexes)
summary_df

# calculate gap ad ratio
summary_df['accuracy_gap'] = (summary_df['Accuracy Valid Report'] - summary_df['Accuracy Train Report']).abs()
summary_df['accuracy_ratio'] = (summary_df['Accuracy Valid Report'] / summary_df['Accuracy Train Report'])
summary_df['log_loss gap'] = (summary_df['Log Loss Valid Report'] - summary_df['Log Loss Train Report']).abs()
summary_df['log_loss ratio'] = (summary_df['Log Loss Valid Report'] / summary_df['Log Loss Train Report'])

summary_df_ok = summary_df.drop(index=('dummy classifier'))

min_index = summary_df_ok['accuracy_gap'].idxmin()
min_row = summary_df_ok.loc[min_index]
print('accuracy gap minimum:','\n' ,min_row)

min_index = summary_df_ok['log_loss gap'].idxmin()
min_row = summary_df_ok.loc[min_index]
print('log loss gap minimum:','\n' ,min_row)

# for visualization - Multinomial Logistic Regression

label_binarizer = LabelBinarizer().fit(y_train)
y_onehot_test = label_binarizer.transform(y_test)
y_onehot_test.shape  # (n_samples, n_classes)

class_of_interest = ["LG6", "LG6A", "MG1", "MG2", "MG3", "MG4", "MG4A", "MG4Zero"]
label_binarizer.transform(["LG6", "LG6A", "MG1", "MG2", "MG3", "MG4", "MG4A", "MG4Zero"])
y_onehot_test = label_binarizer.fit_transform(y_test)

# score calculation - micro_ovr
n_classes = len(class_of_interest)
fpr, tpr, roc_auc = dict(), dict(), dict()
# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_onehot_test.ravel(), y_pred_testproba_mlr.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

print(f"Micro-averaged One-vs-Rest ROC AUC score:\n{roc_auc['micro']:.2f}")

# score calculation - macro_ovr
macro_roc_auc_ovr = roc_auc_score(
    y_test,
    y_pred_testproba_mlr,
    multi_class="ovr",
    average="macro",
)

print(f"Macro-averaged One-vs-Rest ROC AUC score:\n{macro_roc_auc_ovr:.2f}")

from itertools import combinations

pair_list = list(combinations(np.unique(y), 2))
print(pair_list)

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_onehot_test[:, i], y_pred_testproba_mlr[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

fpr_grid = np.linspace(0.0, 1.0, 1000)

# Interpolate all ROC curves at these points
mean_tpr = np.zeros_like(fpr_grid)

for i in range(n_classes):
    mean_tpr += np.interp(fpr_grid, fpr[i], tpr[i])  # linear interpolation

# Average it and compute AUC
mean_tpr /= n_classes

fpr["macro"] = fpr_grid
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

print(f"Macro-averaged One-vs-Rest ROC AUC score:\n{roc_auc['macro']:.2f}")


# create all ROC AUC Curve of Each class in one 
fig, ax = plt.subplots(figsize=(10, 10))

plt.plot(
    fpr["micro"],
    tpr["micro"],
    label=f"micro-average ROC curve (AUC = {roc_auc['micro']:.2f})",
    color="deeppink",
    linestyle=":",
    linewidth=4,
)

plt.plot(
    fpr["macro"],
    tpr["macro"],
    label=f"macro-average ROC curve (AUC = {roc_auc['macro']:.2f})",
    color="navy",
    linestyle=":",
    linewidth=4,
)

colors = cycle(["cyan", "maroon", "orange", "green", "red", "indigo", "purple", "navy"])
for class_id, color in zip(range(n_classes), colors):
    RocCurveDisplay.from_predictions(
        y_onehot_test[:, class_id],
        y_pred_testproba_mlr[:, class_id],
        name=f"ROC curve for {class_of_interest[class_id]}",
        color=color,
        ax=ax,
        plot_chance_level=(class_id == 2),
    )

_ = ax.set(
    xlabel="False Positive Rate",
    ylabel="True Positive Rate",
    title="Extension of Receiver Operating Characteristic\nto One-vs-Rest multiclas of\nMultinomial Logistic Regression",
)

plt.savefig("ROC_AUC_OV_MLR.png")


# Adjust the number of rows and columns
n_cols = 4
n_rows = 2

# Create a figure and a grid of subplots (2 rows and 4 columns)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 10))  # Adjust the figure size if needed
axes = axes.ravel()  # Flatten the 2D array of axes to iterate over it easily

# Loop over each class and its corresponding subplot
for i, class_name in enumerate(class_of_interest):
    class_id = list(label_binarizer.classes_).index(class_name)  # Get the class ID from the binarizer

    # Plot ROC curve in the i-th subplot (axes[i])
    RocCurveDisplay.from_predictions(
        y_onehot_test[:, class_id],    # True binary labels for this class
        y_pred_testproba_mlr[:, class_id],  # Predicted probabilities for this class
        ax=axes[i],                    # Specify which subplot to use
        name=f"{class_name} vs the rest",
        color=plt.cm.get_cmap("tab10")(i),  # Use different colors for each class
        plot_chance_level=True
    )
    axes[i].set_title(f"ROC Curve: {class_name} vs the Rest")  # Set a title for each subplot

# Set the global title, labels, and adjust the layout
fig.suptitle("One-vs-Rest ROC Curves for Each Class", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to not overlap with the title

# Save the figure as a PNG file
plt.savefig("roc_curves_mlr1.png", dpi=600)

# Show the plot (optional)
plt.show()

# LDA Visualization

n_classes = len(class_of_interest)
fpr, tpr, roc_auc = dict(), dict(), dict()
# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_onehot_test.ravel(), y_pred_testproba_lda.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

print(f"Micro-averaged One-vs-Rest ROC AUC score:\n{roc_auc['micro']:.2f}")

macro_roc_auc_ovr = roc_auc_score(
    y_test,
    y_pred_testproba_lda,
    multi_class="ovr",
    average="macro",
)

print(f"Macro-averaged One-vs-Rest ROC AUC score:\n{macro_roc_auc_ovr:.2f}")

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_onehot_test[:, i], y_pred_testproba_lda[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

fpr_grid = np.linspace(0.0, 1.0, 1000)

# Interpolate all ROC curves at these points
mean_tpr = np.zeros_like(fpr_grid)

for i in range(n_classes):
    mean_tpr += np.interp(fpr_grid, fpr[i], tpr[i])  # linear interpolation

# Average it and compute AUC
mean_tpr /= n_classes

fpr["macro"] = fpr_grid
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

print(f"Macro-averaged One-vs-Rest ROC AUC score:\n{roc_auc['macro']:.2f}")

fig, ax = plt.subplots(figsize=(10, 10))

plt.plot(
    fpr["micro"],
    tpr["micro"],
    label=f"micro-average ROC curve (AUC = {roc_auc['micro']:.2f})",
    color="deeppink",
    linestyle=":",
    linewidth=4,
)

plt.plot(
    fpr["macro"],
    tpr["macro"],
    label=f"macro-average ROC curve (AUC = {roc_auc['macro']:.2f})",
    color="navy",
    linestyle=":",
    linewidth=4,
)

colors = cycle(["cyan", "maroon", "orange", "green", "red", "indigo", "purple", "navy"])
for class_id, color in zip(range(n_classes), colors):
    RocCurveDisplay.from_predictions(
        y_onehot_test[:, class_id],
        y_pred_testproba_lda[:, class_id],
        name=f"ROC curve for {class_of_interest[class_id]}",
        color=color,
        ax=ax,
        plot_chance_level=(class_id == 2),
    )

_ = ax.set(
    xlabel="False Positive Rate",
    ylabel="True Positive Rate",
    title="Extension of Receiver Operating Characteristic\nto One-vs-Rest multiclass of\nLinear Discriminant Analysis",
)

plt.savefig("ROC_AUC_OV_LDA.png")

# Adjust the number of rows and columns
n_cols = 4
n_rows = 2

# Create a figure and a grid of subplots (2 rows and 4 columns)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 10))  # Adjust the figure size if needed
axes = axes.ravel()  # Flatten the 2D array of axes to iterate over it easily

# Loop over each class and its corresponding subplot
for i, class_name in enumerate(class_of_interest):
    class_id = list(label_binarizer.classes_).index(class_name)  # Get the class ID from the binarizer

    # Plot ROC curve in the i-th subplot (axes[i])
    RocCurveDisplay.from_predictions(
        y_onehot_test[:, class_id],    # True binary labels for this class
        y_pred_testproba_lda[:, class_id],  # Predicted probabilities for this class
        ax=axes[i],                    # Specify which subplot to use
        name=f"{class_name} vs the rest",
        color=plt.cm.get_cmap("tab10")(i),  # Use different colors for each class
        plot_chance_level=True
    )
    axes[i].set_title(f"ROC Curve: {class_name} vs the Rest")  # Set a title for each subplot

# Set the global title, labels, and adjust the layout
fig.suptitle("One-vs-Rest ROC Curves for Each Class", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to not overlap with the title

# Save the figure as a PNG file
plt.savefig("roc_curves_lda1.png", dpi=600)

# Show the plot (optional)
plt.show()