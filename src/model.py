from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb


def logistic_regression():
    """Instantiate logistic regression object
    Returns:
        model_lr (object): Logistic regression model
    """
    model_lr = LogisticRegression(random_state=42)
    return model_lr

def random_forest():
    """Instantiate random forest object
    Returns:
        model_rf (object): Random forest model
    """
    model_rf = RandomForestClassifier(n_estimators=445, max_depth=16, min_samples_leaf=2, min_samples_split=10, 
                                      random_state=42)
    return model_rf

def xgboost():
    """Instantiate XGboost object
    Returns:
        model_xgb (object): XGBoost model
    """
    model_xgb = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, subsample=1.0, 
                                  n_estimators=200, max_depth=5, learning_rate=0.05, colsample_bytree=0.8)
    return model_xgb