import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor

def create_pipelines():
    """Create model pipelines with StandardScaler."""
    pipelines = {
        "LinearRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", LinearRegression())
        ]),
        "DecisionTree": Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", DecisionTreeRegressor())
        ]),
        "RandomForest": Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", RandomForestRegressor())
        ]),
        "KNeighbors": Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", KNeighborsRegressor())
        ]),
        "XGBoost": Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", XGBRegressor())
        ])
    }
    return pipelines

def train_models(X, y, test_size=0.25, random_state=7):
    """Train all models and return the best performing one."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    pipelines = create_pipelines()
    best_score = float('-inf')
    best_model = None
    
    for name, pipeline in pipelines.items():
        pipeline.fit(X_train, y_train)
        score = pipeline.score(X_test, y_test)
        if score > best_score:
            best_score = score
            best_model = pipeline
            
    return best_model, X_train, X_test, y_train, y_test