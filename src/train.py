import os
import argparse
from azureml.core import Run, Dataset
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import url_parser
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import mean_squared_error,confusion_matrix, precision_score, recall_score, auc,roc_curve
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
import xgboost as xgb
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn import metrics


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    See full source and example: 
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

parser = argparse.ArgumentParser()
parser.add_argument('--regularization', type=float, dest='reg_rate', default=0.01, help='regularization rate')
parser.add_argument("--input-data", type=str, dest='training_dataset_id', help='training dataset')
args = parser.parse_args()

# Set regularization hyperparameter (passed as an argument to the script)
reg = args.reg_rate

# Get the experiment run context
run = Run.get_context()

# Get the training dataset
print("Loading Data...")
malicious_url_df = run.input_datasets['training_data'].to_pandas_dataframe()[['url', 'type']]
malicious_url_df['use_of_ip'] = malicious_url_df['url'].apply(lambda i: url_parser.having_ip_address(i))
malicious_url_df['abnormal_url'] = malicious_url_df['url'].apply(lambda i: url_parser.abnormal_url(i))
malicious_url_df['count.'] = malicious_url_df['url'].apply(lambda i: i.count('.'))
malicious_url_df['count-www'] = malicious_url_df['url'].apply(lambda i: i.count('www'))
malicious_url_df['count@'] = malicious_url_df['url'].apply(lambda i: i.count('@'))
malicious_url_df['count_dir'] = malicious_url_df['url'].apply(lambda i: url_parser.no_of_dir(i))
malicious_url_df['count_embed_domian'] = malicious_url_df['url'].apply(lambda i: url_parser.no_of_embed(i))
malicious_url_df['short_url'] = malicious_url_df['url'].apply(lambda i: url_parser.shortening_service(i))
malicious_url_df['count-https'] = malicious_url_df['url'].apply(lambda i : i.count('https'))
malicious_url_df['count-http'] = malicious_url_df['url'].apply(lambda i : i.count('http'))
malicious_url_df['count%'] = malicious_url_df['url'].apply(lambda i: i.count('%'))
malicious_url_df['count?'] = malicious_url_df['url'].apply(lambda i: i.count('?'))
malicious_url_df['count-'] = malicious_url_df['url'].apply(lambda i: i.count('-'))
malicious_url_df['count='] = malicious_url_df['url'].apply(lambda i: i.count('='))
malicious_url_df['url_length'] = malicious_url_df['url'].apply(lambda i: len(str(i)))
malicious_url_df['hostname_length'] = malicious_url_df['url'].apply(lambda i: len(url_parser.urlparse(i).netloc))
malicious_url_df['sus_url'] = malicious_url_df['url'].apply(lambda i: url_parser.suspicious_words(i))
malicious_url_df['fd_length'] = malicious_url_df['url'].apply(lambda i: url_parser.fd_length(i))
malicious_url_df['tld'] = malicious_url_df['url'].apply(lambda i: url_parser.get_tld(i,fail_silently=True))
malicious_url_df['tld_length'] = malicious_url_df['tld'].apply(lambda i: url_parser.tld_length(i))
malicious_url_df['count-digits']= malicious_url_df['url'].apply(lambda i: url_parser.digit_count(i))
malicious_url_df['count-letters']= malicious_url_df['url'].apply(lambda i: url_parser.letter_count(i))
malicious_url_df = malicious_url_df.drop("tld",1)

print("malicious_url_df type_code",malicious_url_df["type"].value_counts())
lb_make = LabelEncoder()
malicious_url_df["type_code"] = lb_make.fit_transform(
    malicious_url_df["type"]
)
malicious_url_df = malicious_url_df[malicious_url_df["type_code"].isin([2,3,4])]
print("malicious_url_df type_code",malicious_url_df["type_code"].value_counts())

X = malicious_url_df[['use_of_ip','abnormal_url', 'count.', 'count-www', 'count@',
       'count_dir', 'count_embed_domian', 'short_url', 'count-https',
       'count-http', 'count%', 'count?', 'count-', 'count=', 'url_length',
       'hostname_length', 'sus_url', 'fd_length', 'tld_length', 'count-digits',
       'count-letters']]

#Target Variable
y = malicious_url_df['type_code']

X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y, 
    stratify=y, 
    test_size=0.2,
    shuffle=True, 
    random_state=5
)

lgb = LGBMClassifier(objective='multiclass',boosting_type= 'gbdt',n_jobs = 5, 
          silent = True, random_state=5)
LGB_C = lgb.fit(X_train, y_train)


y_pred = LGB_C.predict(X_test)
print(classification_report(y_test,y_pred))

score = metrics.accuracy_score(y_test, y_pred)
print("accuracy:   %0.3f" % score)

cm = metrics.confusion_matrix(y_test, y_pred, labels=[0,1,2,3])
run.log('Confusion Matrix', cm)
plot_confusion_matrix(cm,classes=['benign', 'defacement','phishing','malware'])
# run.log_image("Confusion Matrix Image",plot_confusion_matrix(cm,classes=['benign', 'defacement','phishing','malware']))