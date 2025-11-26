from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

def train_svm(X_train, y_train):
    param_grid = {
        "C": [0.1, 1, 10],
        "kernel": ["linear", "rbf"],
        "gamma": ["scale", "auto"]
    }

    grid = GridSearchCV(
        estimator=SVC(class_weight="balanced", probability=True),
        param_grid=param_grid,
        cv=3,
        scoring="accuracy",
        verbose=2,
        n_jobs=-1
    )

    grid.fit(X_train, y_train)
    print("\nMelhores parâmetros encontrados:", grid.best_params_)

    return grid.best_estimator_
