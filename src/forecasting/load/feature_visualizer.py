import os

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf

from forecasting.load.repository import LoadForecastingRepository


class LoadForecastingFeatureVisualizer:
    """
    Handles exploratory data analysis and plot generation for load forecasting.
    Expects an initialized LoadForecastingRepository.
    """

    def __init__(
        self,
        repository,
        output_dir: str = "saved_graphs",
    ) -> None:
        self.repository: LoadForecastingRepository = repository
        self.output_dir: str = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    async def fetch_and_prepare_data(self) -> pd.DataFrame | None:
        """
        Fetches historical load and weather data from the database,
        aggregates it, and merges it into a clean Pandas DataFrame.
        """
        load_records = await self.repository.get_historical_load_generation()
        weather_records = await self.repository.get_historical_weather_data()

        if not load_records or not weather_records:
            print("Missing data")
            return None

        df_load = pd.DataFrame(
            [
                {"timestamp": record.timestamp, "load_mw": record.load_mw}
                for record in load_records
            ]
        )

        df_weather = pd.DataFrame(
            [
                {
                    "timestamp": record.timestamp,
                    "temperature_2m": record.temperature_2m,
                    "apparent_temperature": record.apparent_temperature,
                    "wind_speed_10m": record.wind_speed_10m,
                    "shortwave_radiation": record.shortwave_radiation,
                    "precipitation": record.precipitation,
                    "relative_humidity_2m": record.relative_humidity_2m,
                }
                for record in weather_records
            ]
        )

        df_load["timestamp"] = pd.to_datetime(df_load["timestamp"], utc=True)
        df_weather["timestamp"] = pd.to_datetime(df_weather["timestamp"], utc=True)

        print("3. Aggregating weather data...")
        df_weather_national = df_weather.groupby("timestamp").mean().reset_index()

        print("4. Merging datasets...")
        df_clean = pd.merge(df_load, df_weather_national, on="timestamp", how="inner")

        df_clean.columns = [
            "timestamp",
            "National Load (MW)",
            "Temp. 2m (°C)",
            "Apparent Temp. (°C)",
            "Wind Speed (km/h)",
            "Solar Rad. (W/m²)",
            "Precipitation (mm)",
            "Humidity (%)",
        ]

        return df_clean

    def generate_correlation_heatmap(self, df_clean: pd.DataFrame) -> None:
        """Generates and saves a Spearman correlation heatmap."""
        print("5. Calculating Spearman correlation and plotting...")

        corr_matrix = df_clean.drop(columns=["timestamp"]).corr(method="spearman")

        sns.set_theme(style="white")
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            corr_matrix,
            mask=mask,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            vmin=-1,
            vmax=1,
            square=True,
            linewidths=0.5,  # type: ignore
            cbar_kws={"shrink": 0.8},
        )

        plt.title(
            "Spearman Correlation Matrix: Energy Load vs. Meteorological Features",
            fontsize=14,
            pad=20,
        )
        plt.xticks(rotation=45, ha="right", fontsize=11)
        plt.yticks(rotation=0, fontsize=11)

        save_path = os.path.join(self.output_dir, "correlation_heatmap.png")
        plt.savefig(save_path, format="png", dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Correlation graph successfully saved in '{save_path}'.")

    def generate_ucurve_hexbin(self, df_clean: pd.DataFrame) -> None:
        """Generates a Hexbin plot showing non-linear dependency."""
        print("Generating U-Curve Hexbin Plot...")

        plt.figure(figsize=(10, 6))

        hb = plt.hexbin(
            df_clean["Apparent Temp. (°C)"],
            df_clean["National Load (MW)"],
            gridsize=50,
            cmap="YlOrRd",
            mincnt=1,
        )

        cb = plt.colorbar(hb, label="Number of Observations")
        cb.set_label("Density (Number of Observations)", fontsize=12)

        z = np.polyfit(
            df_clean["Apparent Temp. (°C)"], df_clean["National Load (MW)"], 2
        )
        p = np.poly1d(z)

        t_min = df_clean["Apparent Temp. (°C)"].min()
        t_max = df_clean["Apparent Temp. (°C)"].max()

        load_at_min = p(t_min)
        load_at_max = p(t_max)

        print(f"Quadratic trend at {t_min:.1f}°C: {load_at_min:.0f} MW")
        print(f"Quadratic trend at {t_max:.1f}°C: {load_at_max:.0f} MW")
        print(f"Asymmetry (winter - summer): {load_at_min - load_at_max:.0f} MW")

        x_trend = np.linspace(
            df_clean["Apparent Temp. (°C)"].min(),
            df_clean["Apparent Temp. (°C)"].max(),
            100,
        )
        plt.plot(
            x_trend, p(x_trend), "b--", linewidth=2, label="Quadratic Trend (U-Curve)"
        )

        plt.title(
            "Non-linear Dependency: National Load vs Apparent Temperature",
            fontsize=14,
            pad=15,
        )
        plt.xlabel("National Apparent Temperature (°C)", fontsize=12)
        plt.ylabel("Aggregated National Load (MW)", fontsize=12)
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.5)

        save_path = os.path.join(self.output_dir, "ucurve_hexbin.png")
        plt.savefig(save_path, format="png", dpi=300, bbox_inches="tight")
        plt.close()
        print(f"U-Curve Plot saved in '{save_path}'.")

    def generate_temporal_boxplots(self, df_clean: pd.DataFrame) -> None:
        """Generates temporal (hourly and weekly) boxplots for load distribution."""
        print("Generating Temporal Boxplots...")

        df_eda = df_clean.copy()
        df_eda["Hour"] = df_eda["timestamp"].dt.hour
        df_eda["Day_of_Week"] = df_eda["timestamp"].dt.day_name()

        df_eda["Day_Type"] = df_eda["timestamp"].dt.dayofweek.apply(
            lambda x: "Weekend" if x >= 5 else "Weekday"
        )

        days_order = [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
        ]

        sns.set_theme(style="whitegrid")
        fig, axes = plt.subplots(1, 2, figsize=(18, 6), sharey=True)

        sns.boxplot(
            data=df_eda,
            x="Hour",
            y="National Load (MW)",
            hue="Day_Type",
            ax=axes[0],
            palette={"Weekday": "#3498db", "Weekend": "#e74c3c"},
            showfliers=False,
        )
        axes[0].set_title("Hourly Load Distribution (Weekday vs. Weekend)", fontsize=14)
        axes[0].set_xlabel("Hour of the Day", fontsize=12)
        axes[0].set_ylabel("National Load (MW)", fontsize=12)
        axes[0].legend(title="Day Type", loc="upper left")

        sns.boxplot(
            data=df_eda,
            x="Day_of_Week",
            y="National Load (MW)",
            ax=axes[1],
            palette="rocket",
            order=days_order,
            showfliers=False,
        )
        axes[1].set_title("Weekly Load Distribution", fontsize=14)
        axes[1].set_xlabel("Day of the Week", fontsize=12)
        axes[1].set_ylabel("")
        axes[1].tick_params(axis="x", rotation=45)

        plt.tight_layout()
        save_path = os.path.join(self.output_dir, "temporal_boxplots.png")
        plt.savefig(save_path, format="png", dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Temporal Boxplots saved in '{save_path}'.")

    def generate_autocorrelation_plot(self, df_clean: pd.DataFrame) -> None:
        """Generates an Autocorrelation (ACF) plot."""
        print("Generating Autocorrelation (ACF) Plot...")

        df_acf = df_clean.sort_values("timestamp")
        series = df_acf["National Load (MW)"]

        lags = 96 * 7

        plt.figure(figsize=(12, 5))

        plot_acf(
            series,
            lags=lags,
            ax=plt.gca(),
            alpha=0.05,
            title="Autocorrelation Function (ACF) of National Load (15-min intervals)",
        )

        tick_positions = np.arange(0, lags + 1, 96)
        tick_labels = [f"Day {i}" for i in range(8)]

        plt.xticks(tick_positions, tick_labels)
        plt.xlabel("Lag (Days)")
        plt.ylabel("Autocorrelation Coefficient")
        plt.grid(axis="y", linestyle="--", alpha=0.7)

        save_path = os.path.join(self.output_dir, "autocorrelation_plot.png")
        plt.savefig(save_path, format="png", dpi=300, bbox_inches="tight")
        plt.close()
        print(f"ACF Plot saved in '{save_path}'.")

    def generate_christmas_transition_multiyear(self, df_clean: pd.DataFrame) -> None:
        """Generates a line plot showing load transition around the Christmas holidays."""
        print("Generating Multi-year Christmas Transition Plot...")

        mask = (
            (df_clean["timestamp"].dt.month == 12)
            & (df_clean["timestamp"].dt.day >= 20)
            & (df_clean["timestamp"].dt.day <= 31)
        )

        df_xmas = df_clean[mask].copy()

        if df_xmas.empty:
            print("No data available for December in the dataset!")
            return

        df_xmas["Year"] = df_xmas["timestamp"].dt.year.astype(str)

        df_xmas["Aligned_Timestamp"] = df_xmas["timestamp"].apply(
            lambda x: x.replace(year=2000)
        )

        plt.figure(figsize=(14, 6))

        sns.lineplot(
            data=df_xmas,
            x="Aligned_Timestamp",
            y="National Load (MW)",
            hue="Year",
            palette="viridis",
            linewidth=1.5,
            alpha=0.8,
        )

        xmas_start = pd.to_datetime("2000-12-25 00:00:00").tz_localize("UTC")
        xmas_end = pd.to_datetime("2000-12-26 23:59:59").tz_localize("UTC")

        plt.axvspan(
            xmas_start,  # type: ignore
            xmas_end,  # type: ignore
            color="#e74c3c",
            alpha=0.2,
            label="Statutory Holidays (Dec 25-26)",
        )

        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        plt.gca().xaxis.set_major_locator(mdates.DayLocator())

        plt.title(
            "Year-over-Year Holiday Inertia: Load Transition around Christmas",
            fontsize=14,
            pad=15,
        )
        plt.xlabel("Date", fontsize=12)
        plt.ylabel("National Load (MW)", fontsize=12)

        plt.legend(
            bbox_to_anchor=(
                0.5,
                -0.25,
            ),
            loc="upper center",
            title="Legend",
            ncol=4,
        )
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.xticks(rotation=45)

        plt.tight_layout()
        save_path = os.path.join(self.output_dir, "christmas_transition_multiyear.png")
        plt.savefig(save_path, format="png", dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Multi-year Christmas Transition Plot saved in '{save_path}'.")

    async def generate_all_visualizations(self) -> None:
        """Helper method to orchestrate fetching data and generating all plots at once."""
        df_clean = await self.fetch_and_prepare_data()

        if df_clean is not None:
            # self.generate_correlation_heatmap(df_clean)
            # self.generate_ucurve_hexbin(df_clean)
            # self.generate_temporal_boxplots(df_clean)
            # self.generate_autocorrelation_plot(df_clean)
            self.generate_christmas_transition_multiyear(df_clean)
