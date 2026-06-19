import os

import pandas as pd
import plotly.graph_objects as go
from matplotlib import pyplot as plt


class LoadForecastingPredictionVisualizer:
    """
    Handles all visual representations of the grid data and model forecasting results.
    Includes built-in methods for saving artifacts to disk.
    """

    def __init__(self, output_dir: str = "output/plots"):
        """
        Initializes the visualizer and ensures the output directory exists.

        Args:
            output_dir (str): Directory where the generated plots will be saved.
        """
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def plot_interactive_timeseries(
        self,
        df: pd.DataFrame,
        x_col: str = "timestamp",
        y_col: str = "load_mw",
        save_name: str = "timeseries.html",
    ):
        """
        Renders and saves a 3-Year Time Series Overview using Plotly.

        Args:
            df (pd.DataFrame): The dataset containing the timeseries.
            x_col (str): The column name for the x-axis (timestamps).
            y_col (str): The column name for the y-axis (values).
            save_name (str): The filename for the saved interactive HTML plot.
        """
        fig = go.Figure()
        fig.add_trace(
            go.Scattergl(
                x=df[x_col],
                y=df[y_col],
                mode="lines",
                name="Value",
                line=dict(width=1, color="#007bff"),
                hovertemplate=(
                    "<b>%{x|%A}</b><br>"
                    + "Date: %{x|%Y-%m-%d %H:%M}<br>"
                    + "Load: %{y} MW<extra></extra>"
                ),
            )
        )

        fig.update_layout(
            title="Time Series Overview",
            template="plotly_white",
            xaxis=dict(rangeslider=dict(visible=True), type="date"),
            yaxis_title="Measurement (MW)",
            height=600,
            hovermode="x unified",
        )

        save_path = os.path.join(self.output_dir, save_name)
        fig.write_html(save_path)
        print(f"Interactive plot saved to {save_path}")
        fig.show()

    def plot_forecast_vs_actual(
        self,
        y_true: list | pd.Series,
        y_pred: list | pd.Series,
        title: str = "ARIMA Baseline Forecast vs Actual",
        save_name: str = "forecast.png",
    ):
        """
        Renders and saves a static baseline plot comparing actuals vs predictions using Matplotlib.

        Args:
            y_true (list | pd.Series): The actual historical values.
            y_pred (list | pd.Series): The values predicted by the model.
            title (str): The title of the plot.
            save_name (str): The filename for the saved PNG image.
        """
        plt.figure(figsize=(12, 6))
        x_axis = range(len(y_true))

        plt.plot(
            x_axis,
            y_true,
            label="Actual Load",
            color="#2c3e50",
            linewidth=2,
            marker="o",
            markersize=4,
        )
        plt.plot(
            x_axis,
            y_pred,
            label="ARIMA Prediction",
            color="#e74c3c",
            linestyle="--",
            linewidth=2,
        )

        plt.title(title, fontsize=14, fontweight="bold")
        plt.xlabel("Interval (15-min steps)", fontsize=12)
        plt.ylabel("Load (MW)", fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.legend()
        plt.tight_layout()

        save_path = os.path.join(self.output_dir, save_name)
        plt.savefig(save_path, dpi=300)
        print(f"Forecast plot saved to {save_path}")
        plt.show()
