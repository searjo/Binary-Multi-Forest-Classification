def evaluate_multi(title, regression, X_train, y, y_pred):
"""The evaluate function will use the get_auc function and then ouput useful information, such as the confusion matrix,
        the classification report.
         Parameters:
                str(title): title of your evaluation in string format
                regression: model being used
                X_train: training data
                y: target data
                y_pred: predicted ys
             Prints: Accuracy, Confusion Matrix, Confusion Matrix with %, Classification Report
             """
    #title is a string denoting the type of data being used (i.e. undersample, oversample)
    regression_acc = regression.score(X_train, y_pred)
    print("Accuracy ({}): {:.2f}%".format(title, regression_acc * 100))
    #regression_acc * 100 because want to see as percentage
    y_pred = regression.predict(X_train)

    #from sklearn.metrics import confusion_matrix, classification_report
    cm = confusion_matrix(y, y_pred)
    clr = classification_report(y, y_pred)
    #classification report is also useful to compare
    
    #plot confusion matrix
    plt.figure(figsize=(8, 8))
    sns.heatmap(cm, annot=True, fmt='g', vmin=0, cmap='Greens')
    #fmt allows strings 
    #vmin sets minimum value of colors to zero
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

    #plot confusion matrix with percentages
    plt.figure(figsize=(8, 8))
    sns.heatmap(cm/np.sum(cm), annot=True,fmt='.2%', cmap='Greens')    
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix with Percentages")
    plt.show()

    print("Classification Report:\n----------------------\n", clr)

