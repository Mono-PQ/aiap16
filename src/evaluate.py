from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score
import pandas as pd
import plotly_express as px

def evaluate(model, X_test, y_test, threshold=0.5, plot=True):
    """Evaluate model based on test data
    Args:
        model (object): trained model object
        X_test (pandas.DataFrame): features used for testing
        y_test (pandas.DataFrame): response used for testing 
        threshold (float, optional): probability threshold to set to determine prediction category. Defaults to 0.5.
    """
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob>=threshold).astype(int)
    TN, FP, FN, TP = confusion_matrix(y_test, y_pred).ravel()
    precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred, average='binary')
    print(f"AUC score: {roc_auc_score(y_test, y_prob):.4f}")
    print(f"Accuracy on Test Set: {accuracy_score(y_test, y_pred) * 100:.4f}%")
    print(f"Confusion Matrix: TN={TN}, FP={FP}, FN={FN}, TP={TP}")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")

    if plot:
        cm = confusion_matrix(y_test, y_pred)
        cm_rf_df = pd.DataFrame(cm, index=[f"Actual {i}" for i in range(len((cm)))],
                            columns=[f'Predicted {i}' for i in range(len(cm))])
        fig = px.imshow(cm_rf_df, labels=dict(x="Predicted Label", y="Actual Label", color="Count"), text_auto=True, 
                        color_continuous_scale='viridis', title="Confusion Matrix")
        fig.write_image("img/confusion_matrix.png")
