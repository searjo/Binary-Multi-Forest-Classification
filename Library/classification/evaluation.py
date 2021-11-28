# From helper functions for ROC and AUC score, if want to use independently
def get_auc(y, y_pred_probabilities, class_labels, column=1, plot=True):
    """Plots ROC AUC
    """
    fpr, tpr, _ = roc_curve(y == column, y_pred_probabilities[:, column], drop_intermediate=False)
    roc_auc = roc_auc_score(y_true=y, y_score=y_pred_probabilities[:, 1])
    print ("AUC: ", roc_auc)
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()


def evaluate(title, regression, X_train, y, y_pred, y_pred_prob):
    """The evaluate function will use the get_auc function and then ouput useful information, such as the confusion matrix,
        the classification report, the precision recall curve, and the roc curve.
         Parameters:
                str(title): title of your evaluation in string format
                regression: model being used
                X_train: training data
                y: target data
                y_pred: predicted ys
                y_pred_prob: predicted probability of y
             Prints: Accuracy, Confusion Matrix, Confusion Matrix with %, Classification Report, Prec/Recall report, AUC

             """
    #Nested get_auc within this function
    def get_auc(y, y_pred_probabilities, class_labels, column=1, plot=True):
        """Plots ROC AUC
        """
        fpr, tpr, _ = roc_curve(y == column, y_pred_probabilities[:, column], drop_intermediate=False)
        roc_auc = roc_auc_score(y_true=y, y_score=y_pred_probabilities[:, 1])
        print ("AUC: ", roc_auc)
        plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.show()
        return get_auc
    # title is a string denoting the type of data being used (i.e. undersample, oversample)
    regression_acc = regression.score(X_train, y_pred)
    print("Accuracy ({}): {:.2f}%".format(title, regression_acc * 100))
    # regression_acc * 100 because want to see as percentage
    y_pred = regression.predict(X_train)

    # from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y, y_pred)
    clr = classification_report(y, y_pred)
    # classification report is also useful to compare

    # plot confusion matrix
    plt.figure(figsize=(8, 8))
    sns.heatmap(cm, annot=True, fmt='g', vmin=0, cmap='Greens')
    # fmt allows strings
    # vmin sets minimum value of colors to zero
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

    # plot confusion matrix with percentages
    plt.figure(figsize=(8, 8))
    sns.heatmap(cm / np.sum(cm), annot=True, fmt='.2%', cmap='Greens')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix with Percentages")
    plt.show()

    print("Classification Report:\n----------------------\n", clr)

    get_auc(y, y_pred_prob, ["Not Target", "Target"], column=1, plot=True)  # Help function

    # Precision recall curve, which is appropriate for an imbalanced data set.
    # from sklearn.metrics import PrecisionRecallDisplay

    display = PrecisionRecallDisplay.from_estimator(
        regression, X_train, y, name="LinearSVC"
    )
    _ = display.ax_.set_title("2-class Precision-Recall curve")

