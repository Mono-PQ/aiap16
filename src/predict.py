def predict(df, model, threshold=0.5):
    """Generate prediction based on data provided
    Args:
        df (pandas.DataFrame): new data for prediction
        model (object): trained model
        threshold (float, optional): probability threshold to set to determine prediction category. Defaults to 0.5.
    Returns:
        y_pred: predicted responses
    """
    y_prob = model.predict_proba(df)[:, 1]
    y_pred = (y_prob>=threshold).astype(int)
    return y_pred