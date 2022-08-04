from xgboost.sklearn import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
import numpy as np
import os
import librosa
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from datetime import datetime
import xgboost as xgb

height = 20
width = 50

mfccPath = ''
modelSavePath = ''
modelSaveLog = ''
STARTED_DATESTRING = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())
modelSaveName = STARTED_DATESTRING + 'trained_model.m'


# Extract MFCC features
def extractMFCC(audio_path, width):
    labels = []
    fileNames = []
    batch_features = []
    flag = 0
    num = 0

    files = os.listdir(audio_path)
    if len(files) > 0:
        lastFileName = files[len(files) - 1]
        for file in files:
            if not file.endswith('.wav'):
                continue
            wave, sr = librosa.load(audio_path + file, mono=True)
            mfcc = librosa.feature.mfcc(wave, sr)
            mfcc = np.pad(mfcc, ((0, 0), (0, width - len(mfcc[0]))), mode='constant', constant_values=0)

            if 'pneumonia' in file:
                label = 1
            else:
                label = 0

            if file[:-22] == filetemp:
                mfcctemp = mfcc + mfcctemp
                flag = flag + 1
                num = num + 1
            else:
                if flag != 0:
                    labeltemp = int(labeltemp)
                    labels.append(labeltemp)
                    fileNames.append(filetemp)
                    mfcctemp = mfcctemp / num
                    batch_features.append(np.array(mfcctemp).T)
                    num = 1
                    flag = flag + 1
                filetemp = file[:-22]
                mfcctemp = mfcc
                labeltemp = label

            if file == lastFileName:
                labeltemp = int(labeltemp)
                labels.append(labeltemp)
                fileNames.append(filetemp)
                mfcctemp = mfcctemp / num
                batch_features.append(np.array(mfcctemp).T)
    else:
        print('Folder is wrong!')

    return np.array(fileNames), np.array(batch_features), np.array(labels)


# Show the accuracy
def show_accuracy(a, b, p):
    acc = a.ravel() == b.ravel()
    print(p + 'Accuracy：%.2f%%' % (100 * float(acc.sum()) / a.size))


# Show precision, recall and f1 score
def printIndex(classifier, x_train, x_test, y_train, y_test):
    y_hat = classifier.predict(x_train)
    show_accuracy(y_hat, y_train, 'training set')  # accuracy
    y_hat = classifier.predict(x_test)
    show_accuracy(y_hat, y_test, 'test set')
    precision = precision_score(y_test, y_hat)  # Precision
    print('Precision:\t{}'.format(precision))
    recall = recall_score(y_test, y_hat)  # Recall
    print('Recall:  \t{}'.format(recall))
    print('f1 score: \t{}'.format(f1_score(y_test, y_hat)))  # f1 score

    y_true, y_pred = y_test, classifier.predict(x_test)
    print(classification_report(y_true, y_pred))


# Tuning parameter: n_estimators
def modelfit(clf, x_train, y_train, x_test, y_test, useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    if useTrainCV:
        xgb_param = clf.get_xgb_params()
        xgtrain = xgb.DMatrix(x_train, label=y_train)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=clf.get_params()['n_estimators'], nfold=cv_folds,
                          metrics='auc', early_stopping_rounds=early_stopping_rounds)
        clf.set_params(n_estimators=cvresult.shape[0])

    print('Best number of trees = {}'.format(cvresult.shape[0]))
    # Fit the algorithm on the data
    clf.fit(x_train, y_train, eval_metric='auc')

    print('Fit on the trainings data')
    clf.fit(x_train, y_train, eval_metric='auc')
    print('Overall AUC:', roc_auc_score(y_train, clf.predict_proba(x_train)[:, 1]))
    print('Predict the probabilities based on features in the test set')
    pred = clf.predict_proba(x_test, ntree_limit=cvresult.shape[0])
    print('Fit on the testings data')
    print('Overall AUC:', roc_auc_score(y_test, clf.predict_proba(x_test)[:, 1]))
    print('Overall AUC:', roc_auc_score(y_test, pred[:, 1]))

    print('Fit on the testings data')
    print('Overall AUC:', roc_auc_score(y_test, clf.predict(x_test)))
    print(' ')


def main():
    # Extract MFCC features
    name, x, y = extractMFCC(mfccPath, width)

    # Split the training set and test set
    name_train, name_test, x_train, x_test, y_train, y_test = train_test_split(name, x, y, test_size=0.2,
                                                                               random_state=23, shuffle=True)

    x_train = x_train.reshape(-1, height * width)
    x_test = x_test.reshape(-1, height * width)

    param_test0 = {'n_estimators': range(5, 100, 5)}
    param_test1 = {
        'max_depth': range(1, 10, 1),
        'min_child_weight': [0, 1, 2]
    }
    param_test3 = {
        'gamma': [i / 10.0 for i in range(0, 5)]
    }
    param_test4 = {
        'subsample': [i / 10.0 for i in range(6, 10)],
        'colsample_bytree': [i / 10.0 for i in range(6, 10)]
    }
    param_test5 = {
        'subsample': [i / 100.0 for i in range(70, 90, 5)],
        'colsample_bytree': [i / 100.0 for i in range(75, 90, 5)]
    }
    param_test6 = {
        'reg_alpha': [1e-5, 1e-2, 0.1, 1, 100]
    }
    search1 = GridSearchCV(estimator=XGBClassifier(learning_rate=0.01, n_estimators=55, max_depth=5,
                                                   min_child_weight=0, gamma=0, subsample=0.7, colsample_bytree=0.6,
                                                   objective='binary:logistic', nthread=4, scale_pos_weight=1,
                                                   reg_alpha=0.2,
                                                   seed=27, use_label_encoder=False),
                           param_grid=param_test4, scoring='roc_auc', n_jobs=4, cv=5)
    search1.fit(x_train, y_train)
    print(search1.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = search1.cv_results_['mean_test_score']
    stds = search1.cv_results_['std_test_score']

    for mean, std, params in zip(means, stds, search1.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))

    xgb1 = XGBClassifier(
        learning_rate=0.01,  # 第0个参数0.1 0.05
        n_estimators=62,  # 第1个参数23 104  5
        max_depth=6,  # 第2个参数  3
        min_child_weight=1,  # 第2个参数
        gamma=0,  # 第3个参数
        subsample=0.86,  # 第4个参数
        colsample_bytree=0.69,  # 第4个参数
        reg_alpha=0.2,
        objective='binary:logistic',
        nthread=4,
        scale_pos_weight=1,
        seed=27,
        use_label_encoder=False)
    modelfit(xgb1, x_train, y_train, x_test, y_test)

    # training classifier
    model = XGBClassifier(max_depth=2, learning_rate=0.3, n_estimators=50, silent=True, reg_lambda=2,
                          objective='binary:logistic')
    model.fit(x_train, y_train)

    # Accuracy of the training set
    x_pred = model.predict(x_train)
    x_train = accuracy_score(y_train, x_pred)
    print('training set accuracy:%2.f%%' % (x_train * 100))

    # Test the classifier
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print('test set accuracy:%2.f%%' % (accuracy * 100))

    # Calculate the AUC
    auc = roc_auc_score(y_test, y_pred)
    print('AUC Score (Test):' + str(auc))

    # Print evaluation metrics
    printIndex(model, x_train, x_test, y_train, y_test)


if __name__ == "__main__":
    main()