from sklearn.model_selection import train_test_split

def split_data(df, test_size=0.3):
    """Split data into training and test sets
    Args:
        df (pandas.DataFrame): cleaned and processed data
        test_size (float, optional): proportion of data to use as test set. Defaults to 0.3.
    Returns:
        X_train, X_test, y_train, y_test (pandas.DataFrame): train and test feature and response data
    """
    X = df.drop(columns=['Lung Cancer Occurrence', 'ID'])
    y = df['Lung Cancer Occurrence']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    return X_train, X_test, y_train, y_test

def train(model, X_train, y_train):
    """Train a logistic regression model and output the model with evaluation metrics
    Args:
        model (object): selected model for training 
        X_train (pandas.Dataframe): feature data used for training
        y_train (pandas.Dataframe): response data used for training
    Returns:
        model (object): trained model object
    """
    model.fit(X_train, y_train)
    return model