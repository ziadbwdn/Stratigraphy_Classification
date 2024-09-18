def accuracy_mx_report(y_true, y_pred, model):
    '''
    function to create confusion matrix output and classification report as an output
    param:
    1. y_true
    2. y_pred
    3. model
    '''
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))
    
    print(f"Confusion Matrix of {model} Method and Accuracy Score Report:\n")
    # Evaluate the accuracy of the LDA model
    model_test_accuracy = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {model_test_accuracy * 100:.2f}% \n")
    # y_test as first argument and the preds as second argument 
    confmtrx_model = confusion_matrix(y_true, y_pred)
    # transform confusion matrix into array
    # the matrix is stored in a vaiable called confmtrx

    # Create DataFrame from confmtrx array 
    # rows for test: y_test as index 
    # columns for preds: y_test_pred_mlr as column
    plt.figure(figsize=(8, 6))
    sns.heatmap(confmtrx_model, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=model.classes_, yticklabels=model.classes_)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()