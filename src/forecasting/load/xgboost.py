import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import cross_val_score


class LoadForecastingXGBoost:
    """
    Wrapper class for training, optimizing, and evaluating XGBoost regression models.
    """

    def __init__(
        self,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        n_estimators: int = 100,
        **kwargs,
    ):
        """
        Initializes the model architecture configured for GPU acceleration.
        """
        self.model_params = {
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "n_estimators": n_estimators,
            "tree_method": "hist",
            "device": "cuda",
            "verbosity": 3,
            **kwargs,
        }
        self.model: xgb.XGBRegressor | None = None

    def optimize_and_train(
        self, y_train, exog_train, cv: int = 3, n_trials: int = 20
    ) -> xgb.XGBRegressor:
        """
        Finds optimal hyperparameters using Optuna for efficient Bayesian optimization.
        """

        def objective(trial):
            param = {
                "max_depth": trial.suggest_int("max_depth", 4, 8),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.01, 0.1, log=True
                ),
                "n_estimators": trial.suggest_int("n_estimators", 300, 1500),
                "tree_method": "hist",
                "device": "cuda",
                "verbosity": 3,
            }

            trial_model = xgb.XGBRegressor(**param)

            scores = cross_val_score(
                trial_model,
                exog_train,
                y_train,
                cv=cv,
                scoring="neg_mean_absolute_error",
                n_jobs=1,
            )
            return -scores.mean()

        optuna.logging.set_verbosity(optuna.logging.DEBUG)
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials)

        self.model_params.update(study.best_params)
        print(f"Optimal parameters found: {study.best_params}")
        print(f"Best cross-validation MAE: {study.best_value:.4f}")

        return self.train(y_train, exog_train)

    def train(
        self, y_train: pd.Series, exog_train: pd.DataFrame, **fit_kwargs
    ) -> xgb.XGBRegressor:
        """
        Fits the XGBoost Regressor using the configured GPU parameters.
        """
        self.model = xgb.XGBRegressor(**self.model_params)
        self.model.fit(exog_train, y_train, **fit_kwargs)
        return self.model

    def summary(self) -> str:
        """Returns feature importance metrics as a pseudo-summary."""
        if self.model is None:
            return "Model must be trained before accessing summary statistics."

        importances = self.model.feature_importances_
        return f"XGBoost Model summary initialized successfully.\nHyperparameters: {self.model_params}\nTop Feature Importances Raw: {importances[:10]}"

    def forecast(self, steps: int, exog_forecast: pd.DataFrame):
        """
        Generates predictions for the specified out-of-sample data grid.
        """
        if self.model is None:
            raise ValueError("Model must be trained before forecasting.")

        if len(exog_forecast) != steps:
            exog_forecast = exog_forecast.iloc[:steps]

        preds = self.model.predict(exog_forecast)
        return pd.Series(preds, index=exog_forecast.index)

    def evaluate(self, y_true, y_pred) -> dict:
        """
        Calculates standard regression metrics to evaluate model performance.
        """
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-9))) * 100
        return {"RMSE": rmse, "MAE": mae, "MAPE": round(mape, 2)}
