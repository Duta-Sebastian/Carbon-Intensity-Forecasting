import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure


class ForecastPlotter:
    """Dedicated engine for generating ML visualization charts."""

    @staticmethod
    def plot_train_test_validation(
        train: pd.Series,
        test: pd.Series,
        predictions: pd.Series,
        model_name: str,
        country_code: str,
        metrics: dict[str, float],
    ) -> Figure:
        """Generates a standardized validation plot for time series forecasting."""
        fig, ax = plt.subplots(figsize=(18, 9))

        ax.plot(train.index, train, label="Training (80%)", color="gray", alpha=0.4)
        ax.plot(test.index, test, label="Actual (20% Test)", color="green", linewidth=2)
        ax.plot(
            predictions.index,
            predictions,
            label=f"{model_name} Prediction",
            color="red",
            linestyle="--",
        )

        title = (
            f"{model_name} Wind Forecast Validation ({country_code})\n"
            f"MSE: {metrics['mse']:.2f} | RMSE: {metrics['rmse']:.2f} | MAE: {metrics['mae']:.2f}"
        )
        ax.set_title(title, fontsize=16, fontweight="bold")
        ax.set_xlabel("Time (UTC)", fontsize=12)
        ax.set_ylabel("Generation (MW)", fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        return fig
