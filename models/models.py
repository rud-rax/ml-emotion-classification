from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
from xgboost import XGBClassifier  # Requires xgboost
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import make_scorer, f1_score

# Define stratified k-fold cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Define a custom scoring function (F1-score)
custom_scorer = make_scorer(f1_score, average='weighted')

# Define models and their hyperparameter search spaces
models = {
    "LogisticRegression": (
        LogisticRegression(max_iter=1000),
        {
            'C': Real(0.01, 10, prior='log-uniform'),
            'penalty': Categorical(['l1', 'l2']),
            'solver': Categorical(['liblinear', 'saga'])
        }
    ),
    "RandomForest": (
        RandomForestClassifier(),
        {
            'n_estimators': Integer(50, 500),
            'max_depth': Integer(3, 20),
            'min_samples_split': Integer(2, 10),
            'min_samples_leaf': Integer(1, 5)
        }
    ),
    "GradientBoosting": (
        GradientBoostingClassifier(),
        {
            'n_estimators': Integer(100, 500),
            'learning_rate': Real(0.01, 0.3),
            'max_depth': Integer(3, 10),
            'min_samples_split': Integer(2, 10),
            'min_samples_leaf': Integer(1, 5)
        }
    ),
    "XGBoost": (
        XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
        {
            'n_estimators': Integer(100, 500),
            'learning_rate': Real(0.01, 0.3),
            'max_depth': Integer(3, 10),
            'subsample': Real(0.5, 1.0),
            'colsample_bytree': Real(0.5, 1.0)
        }
    ),
    "SVM": (
        SVC(),
        {
            'C': Real(0.01, 10, prior='log-uniform'),
            'kernel': Categorical(['linear', 'poly', 'rbf', 'sigmoid']),
            'gamma': Real(0.0001, 1, prior='log-uniform')
        }
    ),
    "KNN": (
        KNeighborsClassifier(),
        {
            'n_neighbors': Integer(3, 20),
            'weights': Categorical(['uniform', 'distance']),
            'metric': Categorical(['euclidean', 'manhattan', 'minkowski'])
        }
    )
}
