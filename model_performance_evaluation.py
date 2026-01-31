#------ This file defines the functions used mode model evaluation -----------
#Import libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score,roc_curve,confusion_matrix,classification_report, auc, f1_score
import seaborn as sns
from pylab import rcParams

#Function definitions
def plot_conf_matrix(y_test,y_preds):
    conf_mat = confusion_matrix(y_test,y_preds)
    sns.heatmap(conf_mat, annot=True,cmap="Blues",fmt="g")
    plt.xlabel("Predicted values")
    plt.ylabel("Actual Values")
    plt.title("Confusion Matrix")

def to_labels (pos_probs,threshold):
    return(pos_probs>=threshold).astype('int') #return 0 or 1

def evaluateFinalModel(y_test,y_pred_proba_class1,y_preds):
    #define metrics
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba_class1)
    auc_1 = roc_auc_score(y_test, y_pred_proba_class1)
    auc_2 = auc(fpr, tpr)
    print(f"AUC {auc_2:0.2f}")

    #best threshold in ROC
    # gmeans = np.sqrt(tpr*(1-fpr))
    # ix = np.argmax(gmeans)

    rcParams['figure.figsize'] = 5,5
    plt.rcParams.update({'font.size': 16})
    plt.rcParams.update({'figure.dpi':300})

    #create ROC curve
    plt.plot(fpr,tpr,label="AUC="+str(round(auc_1,2)))
    # plt.scatter(fpr[ix],tpr[ix],label='Optimal threshold {%0.5f}'%t[ix], marker='o', color='black')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(loc=4)
    plt.show()

    #Confusion matrix
    plot_conf_matrix(y_test, y_preds)

    #Classification Report
    cls_report_final = classification_report(y_test,y_preds)
    print(cls_report_final)

def tune_threshold(y_test,y_pred_proba_class1,y_preds):
    #Tuning threshold
    threshold_range = np.arange(0,1,0.001)
    scores = [f1_score(y_test, to_labels(y_pred_proba_class1, threshold)) for threshold in threshold_range]
    ix_t = np.argmax(scores)
    best_t = threshold_range[ix_t]
    best_f1_score = scores[ix_t]
    return best_t,best_f1_score