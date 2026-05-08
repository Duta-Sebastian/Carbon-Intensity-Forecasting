from datetime import datetime
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

from core.types import CountryCode
from forecasting.wind.repository import WindForecastingRepository


class WindGraphingService:
    """
    Service responsible for fetching raw wind data and generating visual plots.
    """

    def __init__(self, repository: WindForecastingRepository) -> None:
        self.repository = repository

    async def generate_historical_plot(
        self,
        country_code: CountryCode,
        start_date: datetime,
        end_date: datetime,
        save_path: Optional[str] = None,
        plot_median: bool = False,
        plot_average: bool = False,
    ) -> Figure | str:
        """
        Fetches historical wind data and plots it over the specified time period.
        Optionally saves the plot to a file if `save_path` is provided.
        """
        records = await self.repository.get_historical_wind_generation(
            country_code=country_code,
            start_date=start_date,
            end_date=end_date,
        )

        if not records:
            raise ValueError(
                f"No wind generation data found for {country_code.value} in the given time period."
            )

        df = pd.DataFrame(
            [
                {"timestamp": record.timestamp, "generation_mw": record.generation_mw}
                for record in records
            ]
        )

        df.sort_values("timestamp", inplace=True)
        df.set_index("timestamp", inplace=True)

        fig, ax = plt.subplots(figsize=(16, 8))

        ax.plot(
            df.index,
            df["generation_mw"],
            color="#2ca02c",
            linewidth=1.5,
            label="Onshore Wind Generation (MW)",
        )

        if plot_average:
            avg_mw = df["generation_mw"].mean()
            ax.axhline(
                y=avg_mw,
                color="#d62728",
                linestyle="--",
                linewidth=2,
                label=f"Average: {avg_mw:.1f} MW",
            )

        if plot_median:
            median_mw = df["generation_mw"].median()
            ax.axhline(
                y=median_mw,
                color="#1f77b4",
                linestyle=":",
                linewidth=2,
                label=f"Median: {median_mw:.1f} MW",
            )

        ax.set_title(
            f"Historical Onshore Wind Generation ({country_code.value})\n"
            f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
            fontsize=16,
            fontweight="bold",
        )
        ax.set_xlabel("Time (UTC)", fontsize=14)
        ax.set_ylabel("Power Generation (MW)", fontsize=14)
        ax.grid(True, linestyle="--", alpha=0.7)

        ax.legend(fontsize=12)

        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300)
            plt.close(fig)
            return save_path

        return fig
