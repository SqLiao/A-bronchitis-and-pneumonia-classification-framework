import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt
import os
import librosa
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, roc_curve, auc
import joblib
from datetime import datetime

height = 20
width = 50

mfccPath = ''
modelSavePath = ''
modelSaveLog = ''
STARTED_DATESTRING = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())
modelSaveName = STARTED_DATESTRING + 'trained_model.m'
fromSaveName = ''
ROCSavePath = ''


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


# Save the trained model
def saveSVM(classifier, p, saveName):
    os.chdir(p)
    saveLog = os.path.join(modelSaveLog, saveName)
    joblib.dump(classifier, saveLog)


# Load the trained model
def loadSVM(p, loadName):
    os.chdir(p)
    clf_linear = joblib.load(loadName)
    return clf_linear


# Automatic tuning parameters
def autoPara(x_train, x_test, y_train, y_test):
    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4, 1e-5, 1e-6, 1e-7],
                         'C': [0.1, 1, 10, 100, 1000, 10000]},
                        {'kernel': ['linear'], 'C': [0.1, 1, 10, 100, 1000, 10000]}]
    scores = ['precision', 'recall']
    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()

        # 调用 GridSearchCV，将 SVC(), tuned_parameters, cv=5, 还有 scoring 传递进去，
        clf = GridSearchCV(svm.SVC(), tuned_parameters, cv=10,
                           scoring='%s_macro' % score)  # cv为迭代次数。#基于交叉验证的网格搜索，cv:确定交叉验证拆分策略。
        # 用训练集训练这个学习器 clf
        clf.fit(x_train, y_train)

        print("Best parameters set found on development set:")
        print()

        # 再调用 clf.best_params_ 就能直接得到最好的参数搭配结果
        print(clf.best_params_)

        print()
        print("Grid scores on development set:")
        print()
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']

        # 看一下具体的参数间不同数值的组合后得到的分数是多少
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean, std * 2, params))

        print()

        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        y_true, y_pred = y_test, clf.predict(x_test)

        # 打印在测试集上的预测结果与真实值的分数
        print(classification_report(y_true, y_pred))

        print()


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


# Compute ROC curve and ROC area for each class
def pltROC(y_test, test_y_score):
    fpr, tpr, threshold = roc_curve(y_test, test_y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color='darkorange',
             lw=3, label='SVM (AUC = %0.2f)' % roc_auc)
    plt.scatter(fpr[4], tpr[4], color='black', lw=3, marker='s')
    plt.plot([0, fpr[4]], [tpr[4], tpr[4]], color='black', lw=2, ls='--')
    plt.plot([fpr[4], fpr[4]], [0, tpr[4]], color='black', lw=2, ls='--')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.tick_params(labelsize=15)
    plt.xlabel('Sensitivity', size=15)
    plt.ylabel('Specificity', size=15)
    plt.title('SVM grouping without augmentation', size=15)
    plt.legend(loc='lower right', prop={'size': 15})
    plt.savefig(ROCSavePath + 'ROC.pdf', dpi=1200)
    plt.show()


def main():
    # Extract MFCC features
    name, x, y = extractMFCC(mfccPath, width)

    # Split the training set and test set
    name_train, name_test, x_train, x_test, y_train, y_test = train_test_split(name, x, y, test_size=0.2,
                                                                               random_state=23, shuffle=True)

    x_train = x_train.reshape(-1, height * width)
    x_test = x_test.reshape(-1, height * width)

    # Automatic tuning parameters
    # autoPara(x_train,x_test,y_train,y_test)

    # The optimal hyper-parameters are obtained by automatic parameter tuning: kernel='rbf', C=10, gamma = 1e-6
    # training classifier
    clf = svm.SVC(kernel='rbf', C=10, gamma=1e-6, probability=True)
    model = clf.fit(x_train, y_train)

    # Load the trained model
    # clf = loadSVM(modelSaveLog, fromSaveName)

    # Test the classifier
    y_hat = clf.predict(x_test)
    print('test data:', end='')
    print(name_test)
    print('The actual class:', end='')
    print(y_test)
    print('The test class:', end='')
    print(y_hat)

    # Plot ROC curve
    test_y_score = model.decision_function(x_test)
    pltROC(y_test, test_y_score)

    # Print evaluation metrics
    printIndex(clf, x_train, x_test, y_train, y_test)

    # Save the trained model
    saveSVM(clf, modelSavePath, modelSaveName)


if __name__ == "__main__":
    main()
