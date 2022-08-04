import numpy as np
from sklearn.ensemble import RandomForestClassifier
import librosa
import os
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV
from XGBoost import printIndex

height = 20
width = 50

mfccPath = ''


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


# Automatic tuning parameters
def arg_opi(X_train, y_train, X_test, y_test):
    param_test1 = {'n_estimators': range(5, 100, 5)}
    search1 = GridSearchCV(estimator=RandomForestClassifier(min_samples_split=5,
                                                            min_samples_leaf=2,
                                                            max_depth=45, random_state=12),
                           param_grid=param_test1,
                           scoring='roc_auc',
                           cv=5)

    # param_test2 = {'min_samples_split': range(1, 20, 1), 'min_samples_leaf': range(1, 20, 1)}
    # search1 = GridSearchCV(estimator=RandomForestClassifier(n_estimators=39,
    #                                                          max_depth=5, random_state=12),
    #                         param_grid=param_test2,
    #                         scoring='roc_auc',
    #                         cv=5)
    #
    # param_test3 = {'max_depth': range(1, 10, 1)}
    # search1 = GridSearchCV(estimator=RandomForestClassifier(min_samples_split=3,
    #                                                          min_samples_leaf=1,
    #                                                          n_estimators=95,
    #                                                          random_state=12),
    #                         param_grid=param_test3,
    #                         scoring='roc_auc',
    #                         cv=5)
    #
    # param_test4 = {'criterion': ['gini', 'entropy'], 'class_weight': [None, 'balanced']}
    # search1 = GridSearchCV(estimator=RandomForestClassifier(n_estimators=90,
    #                                                          min_samples_split=8,
    #                                                          min_samples_leaf=2,
    #                                                          max_depth=8,
    #                                                          random_state=12,
    #                                                          ),
    #                         param_grid=param_test4,
    #                         scoring='roc_auc',
    #                         cv=5)

    search1.fit(X_train, y_train)
    print(search1.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = search1.cv_results_['mean_test_score']
    stds = search1.cv_results_['std_test_score']

    # 看一下具体的参数间不同数值的组合后得到的分数是多少
    for mean, std, params in zip(means, stds, search1.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))

    # 整合所有最优参数值，得到最优评分
    best_score = roc_auc_score(y_test, search1.best_estimator_.predict_proba(X_test)[:, 1])
    print(best_score)


def main():
    # Extract MFCC features
    name, x, y = extractMFCC(mfccPath, width)

    # Split the training set and test set
    name_train, name_test, X_train, X_test, y_train, y_test = train_test_split(name, x, y, test_size=0.2,
                                                                               random_state=23, shuffle=True)

    X_train = X_train.reshape(-1, height * width)
    X_test = X_test.reshape(-1, height * width)

    # Automatic tuning parameters
    arg_opi(X_train, y_train, X_test, y_test)

    rfc = RandomForestClassifier(random_state=12,
                                 n_estimators=10,
                                 min_samples_split=7,
                                 min_samples_leaf=3,
                                 max_depth=5,
                                 criterion='gini',
                                 class_weight=None)
    rfc.fit(X_train, y_train)
    result = rfc.score(X_test, y_test)
    print(result)

    print('Fit on the trainings data')
    print('Overall AUC:', roc_auc_score(y_train, rfc.predict_proba(X_train)[:, 1]))

    print('Fit on the testings data')
    print('Overall AUC:', roc_auc_score(y_test, rfc.predict_proba(X_test)[:, 1]))

    # Print evaluation metrics
    printIndex(rfc, X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    main()
