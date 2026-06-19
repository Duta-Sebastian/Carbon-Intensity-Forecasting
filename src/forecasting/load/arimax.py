import numpy as np
import pandas as pd
import pmdarima as pm
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.iolib.summary import Summary
from statsmodels.tsa.statespace.sarimax import SARIMAX as ARIMA
from statsmodels.tsa.statespace.sarimax import (
    SARIMAXResultsWrapper as ARIMAResultsWrapper,
)


class LoadForecastingARIMAX:
    """
    Wrapper class for training, optimizing, and evaluating ARIMAX/SARIMAX models.
    """

    def __init__(self, order: tuple = (1, 1, 1), seasonal_order: tuple = (0, 0, 0, 0)):
        """
        Initializes the model architecture.

        Args:
            order (tuple): The (p, d, q) order of the model for the number of AR parameters,
                           differences, and MA parameters.
            seasonal_order (tuple): The (P, D, Q, s) order of the seasonal component.
        """
        self.order = order
        self.seasonal_order = seasonal_order
        self.model_res: ARIMAResultsWrapper | None = None

    def optimize_and_train(
        self, y_train, exog_train=None, max_p: int = 5, max_q: int = 5
    ) -> ARIMAResultsWrapper:
        """
        Finds the optimal (p, d, q) parameters using pmdarima,
        then trains the final statsmodels instance.

        Args:
            y_train: The endogenous target variable.
            exog_train: The exogenous features matrix (e.g., Fourier harmonics).
            max_p (int): Maximum AutoRegressive degree to search.
            max_q (int): Maximum Moving Average degree to search.

        Returns:
            ARIMAResultsWrapper: The fitted statsmodels results object.
        """
        auto_model = pm.auto_arima(
            y=y_train,
            X=exog_train,
            start_p=0,
            max_p=max_p,
            start_q=0,
            max_q=max_q,
            d=None,
            seasonal=False,
            stepwise=True,
            trace=True,
            error_action="ignore",
            suppress_warnings=True,
        )

        self.order = auto_model.order
        return self.train(y_train, exog_train)

    def train(
        self,
        y_train: pd.Series,
        exog_train: pd.DataFrame | None = None,
        method: str = "lbfgs",
        maxiter: int = 500,
        **fit_kwargs,
    ) -> ARIMAResultsWrapper:
        """
        Fits the SARIMAX state-space model to the provided time-series data.

        This method orchestrates the Maximum Likelihood Estimation (MLE) process,
        optimizing model parameters for the specified endogenous target and
        exogenous regressor matrix. It enforces stationarity and invertibility
        constraints to ensure numerical stability.

        Args:
            y_train (pd.Series): The endogenous target time-series (e.g., load_mw).
            exog_train (pd.DataFrame, optional): The exogenous regressor matrix
                representing external drivers (e.g., weather/seasonal harmonics).
            method (str, optional): The optimization algorithm (e.g., 'lbfgs', 'cg', 'nm').
                Defaults to 'lbfgs'.
            maxiter (int, optional): The maximum number of iterations allowed for
                the optimizer to achieve convergence. Defaults to 500.
            **fit_kwargs: Arbitrary keyword arguments passed directly to the
                underlying statsmodels.fit() method (e.g., 'tol', 'disp').

        Returns:
            SARIMAXResultsWrapper: The fitted results object containing model
                coefficients, residual diagnostics, and convergence metadata.

        Raises:
            ConvergenceWarning: Logged if the model reaches maxiter without
                satisfying the convergence tolerance, indicating the optimization
                surface may be ill-conditioned or the model order too complex.
        """
        model = ARIMA(
            y_train,
            exog=exog_train,
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=True,
            enforce_invertibility=True,
        )

        self.model_res = model.fit(
            method=method, maxiter=maxiter, disp=True, **fit_kwargs
        )  # type: ignore
        return self.model_res  # type: ignore

    def summary(self) -> Summary | None:
        """Returns the statistical summary of the fitted model."""
        if self.model_res is None:
            return None
        return self.model_res.summary()

    def forecast(self, steps: int, exog_forecast=None):
        """
        Generates out-of-sample predictions for the specified number of steps.

        Args:
            steps (int): The number of future time steps to forecast.
            exog_forecast: The exogenous features matrix for the future horizon.

        Returns:
            pd.Series: The forecasted values.
        """
        if self.model_res is None:
            raise ValueError("Model must be trained before forecasting.")
        return self.model_res.forecast(steps=steps, exog=exog_forecast)

    def evaluate(self, y_true, y_pred) -> dict:
        """
        Calculates standard regression metrics to evaluate model performance.

        Args:
            y_true: The ground truth values.
            y_pred: The forecasted values.

        Returns:
            dict: A dictionary containing RMSE, MAE, and MAPE.
        """
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-9))) * 100
        return {"RMSE": rmse, "MAE": mae, "MAPE": round(mape, 2)}
