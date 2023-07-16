
import numpy as np

from sklearn.metrics import f1_score

from sklearn.model_selection import cross_validate
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn import tree
import itertools
import timeit
from matplotlib import pyplot as plt

from sklearn.metrics import mean_absolute_error
from pandas.api.types import is_object_dtype, is_numeric_dtype

class visualization:
  
  @staticmethod
  def plotScoreVSDepth(max_depth, f1_score_test, f1_score_train, title):
    print(" f1 score vs max depth of the tree ")
    plt.plot(max_depth, f1_score_test,  'o-', color = 'r', label='Test F1 Score')
    plt.plot(max_depth, f1_score_train, 'o-', color = 'b', label='Train F1 Score')
    plt.xlabel('Max Tree Depth')
    plt.ylabel('Model F1 Score')
   
    plt.title(title)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()

  @staticmethod
  def plotDecisionTree(clf, feature_names):
    fig = plt.figure(figsize=(50,40))
    _ = tree.plot_tree(clf, 
                   feature_names=feature_names,  
                   #class_names={0:'Malignant', 1:'Benign'},
                   filled=True,
                  fontsize=15)
    
  @staticmethod
  def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues):

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(2), range(2)):
        plt.text(j, i, format(cm[i, j], fmt),
             horizontalalignment="center",
             color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')        
    plt.show()    

  @staticmethod
  def model_evaluation(y_valid, y_pred):
    
    auc = roc_auc_score(y_valid, y_pred) if not (y_valid.dtypes[0] == object) else 0
    f1 = f1_score(y_valid,y_pred, pos_label='yes' if y_valid.dtypes[0] == object else 1) # for numerical datatypes should be 1, and 'yes' for object dtypes.
    accuracy = accuracy_score(y_valid,y_pred)
    precision = precision_score(y_valid,y_pred, pos_label='yes' if y_valid.dtypes[0] == object else 1)
    recall = recall_score(y_valid,y_pred, pos_label='yes' if y_valid.dtypes[0] == object else 1)
    
    print("mean abs error:  "+"{:.2f}".format(mean_absolute_error(y_valid, y_pred)  if not (y_valid.dtypes[0] == object) else 0  ))
    print("F1 Score:  "+"{:.2f}".format(f1))
    print("Accuracy:  "+"{:.2f}".format(auc)+"     AUC:          "+"{:.2f}".format(auc))
    print("Accuracy:  "+"{:.2f}".format(accuracy)+"  Accuracy:     "+"{:.2f}".format(accuracy))
    print("Precision: "+"{:.2f}".format(precision)+"  Precision: "+"{:.2f}".format(precision))
    print("Precision: "+"{:.2f}".format(recall)+"     Recall:    "+"{:.2f}".format(recall))

  @staticmethod
  def plot_learning_curve(clf, X, y, title="Insert Title"):
    
    nn = len(y)
    train_mean = []; train_std = [] #model performance score (f1)
    cv_mean = []; cv_std = [] #model performance score (f1) cross validation
    fit_mean = []; fit_std = [] #model fit/training time
    pred_mean = []; pred_std = [] #model test/prediction times
    train_sizes=(np.linspace(.05, 1.0, 10)*nn).astype('int')  
    print(train_sizes)
    
    for i in train_sizes:
        idx = np.random.randint(X.shape[0], size=i)
        print(len(X))
        print(idx)
        X_subset = X[idx,:]
        y_subset = y[idx]
        scores = cross_validate(clf, X_subset, y_subset, cv=10, scoring='f1', n_jobs=-1, return_train_score=True)
        
        train_mean.append(np.mean(scores['train_score'])); train_std.append(np.std(scores['train_score']))
        cv_mean.append(np.mean(scores['test_score'])); cv_std.append(np.std(scores['test_score']))
        fit_mean.append(np.mean(scores['fit_time'])); fit_std.append(np.std(scores['fit_time']))
        pred_mean.append(np.mean(scores['score_time'])); pred_std.append(np.std(scores['score_time']))
    
    train_mean = np.array(train_mean); train_std = np.array(train_std)
    cv_mean = np.array(cv_mean); cv_std = np.array(cv_std)
    fit_mean = np.array(fit_mean); fit_std = np.array(fit_std)
    pred_mean = np.array(pred_mean); pred_std = np.array(pred_std)
    
    visualization.plot_LC(train_sizes, train_mean, train_std, cv_mean, cv_std, title)
    visualization.plot_times(train_sizes, fit_mean, fit_std, pred_mean, pred_std, title)
    
    return train_sizes, train_mean, fit_mean, pred_mean
  
  
  @staticmethod
  def final_classifier_evaluation(clf,X_train, x_valid, y_train, y_valid):
    
    start_time = timeit.default_timer()
    clf.fit(X_train, y_train)
    end_time = timeit.default_timer()
    training_time = end_time - start_time
    
    start_time = timeit.default_timer()    
    y_pred = clf.predict(x_valid)
    end_time = timeit.default_timer()
    pred_time = end_time - start_time
    
    auc = roc_auc_score(y_valid, y_pred) if not (y_valid.dtypes[0] == object) else 0
    f1 = f1_score(y_valid,y_pred, pos_label='yes' if y_valid.dtypes[0] == object else 1)
    accuracy = accuracy_score(y_valid,y_pred)
    precision = precision_score(y_valid,y_pred, pos_label='yes' if y_valid.dtypes[0] == object else 1)
    recall = recall_score(y_valid,y_pred, pos_label='yes' if y_valid.dtypes[0] == object else 1)
    cm = confusion_matrix(y_valid,y_pred)

    print("Model Evaluation Metrics Using Untouched Test Dataset")
    print("*****************************************************")
    print("Model Training Time (s):   "+"{:.5f}".format(training_time))
    print("Model Prediction Time (s): "+"{:.5f}\n".format(pred_time))
    print("F1 Score:  "+"{:.2f}".format(f1))
    print("Accuracy:  "+"{:.2f}".format(accuracy)+"     AUC:       "+"{:.2f}".format(auc))
    print("Precision: "+"{:.2f}".format(precision)+"     Recall:    "+"{:.2f}".format(recall))
    print("*****************************************************")
    plt.figure()
    visualization.plot_confusion_matrix(cm, classes=["0","1"], title='Confusion Matrix')
    plt.show()


  @staticmethod
  def plot_LC(train_sizes, train_mean, train_std, cv_mean, cv_std, title):
    
    plt.figure()
    plt.title("Learning Curve: "+ title)
    plt.xlabel("Training Examples")
    plt.ylabel("Model F1 Score")
    plt.fill_between(train_sizes, train_mean - 2*train_std, train_mean + 2*train_std, alpha=0.1, color="b")
    plt.fill_between(train_sizes, cv_mean - 2*cv_std, cv_mean + 2*cv_std, alpha=0.1, color="r")
    plt.plot(train_sizes, train_mean, 'o-', color="b", label="Training Score")
    plt.plot(train_sizes, cv_mean, 'o-', color="r", label="Cross-Validation Score")
    plt.legend(loc="best")
    plt.show()
    
  @staticmethod  
  def plot_times(train_sizes, fit_mean, fit_std, pred_mean, pred_std, title):
    
    plt.figure()
    plt.title("Modeling Time: "+ title)
    plt.xlabel("Training Examples")
    plt.ylabel("Training Time (s)")
    plt.fill_between(train_sizes, fit_mean - 2*fit_std, fit_mean + 2*fit_std, alpha=0.1, color="b")
    plt.fill_between(train_sizes, pred_mean - 2*pred_std, pred_mean + 2*pred_std, alpha=0.1, color="r")
    plt.plot(train_sizes, fit_mean, 'o-', color="b", label="Training Time (s)")
    plt.plot(train_sizes, pred_std, 'o-', color="r", label="Prediction Time (s)")
    plt.legend(loc="best")
    plt.show()    
    plt.figure()
    plt.title("Learning Curve: "+ title)
    plt.xlabel("Training Examples")
    plt.ylabel("Model F1 Score")
    plt.fill_between(train_sizes, train_mean - 2*train_std, train_mean + 2*train_std, alpha=0.1, color="b")
    plt.fill_between(train_sizes, cv_mean - 2*cv_std, cv_mean + 2*cv_std, alpha=0.1, color="r")
    plt.plot(train_sizes, train_mean, 'o-', color="b", label="Training Score")
    plt.plot(train_sizes, cv_mean, 'o-', color="r", label="Cross-Validation Score")
    plt.legend(loc="best")
    plt.show()
    
  @staticmethod
  def plot_times(train_sizes, fit_mean, fit_std, pred_mean, pred_std, title):
    
    plt.figure()
    plt.title("Modeling Time: "+ title)
    plt.xlabel("Training Examples")
    plt.ylabel("Training Time (s)")
    plt.fill_between(train_sizes, fit_mean - 2*fit_std, fit_mean + 2*fit_std, alpha=0.1, color="b")
    plt.fill_between(train_sizes, pred_mean - 2*pred_std, pred_mean + 2*pred_std, alpha=0.1, color="r")
    plt.plot(train_sizes, fit_mean, 'o-', color="b", label="Training Time (s)")
    plt.plot(train_sizes, pred_std, 'o-', color="r", label="Prediction Time (s)")
    plt.legend(loc="best")
    plt.show()    


  @staticmethod
  def dictionary(results):
    plt.plot(list(results.keys()), list(results.values()))
    plt.show()