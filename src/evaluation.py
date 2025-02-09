from sklearn import metrics
import numpy as np

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance using multiple metrics."""
    pred = model.predict(X_test)
    
    metrics_dict = {
        "R^2": metrics.r2_score(y_test, pred),
        "Adjusted R^2": 1 - (1-metrics.r2_score(y_test, pred))*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1),
        "MAE": metrics.mean_absolute_error(y_test, pred),
        "MSE": metrics.mean_squared_error(y_test, pred),
        "RMSE": np.sqrt(metrics.mean_squared_error(y_test, pred))
    }
    
    return metrics_dict