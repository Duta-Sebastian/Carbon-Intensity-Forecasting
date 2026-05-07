from typing import Any, Literal, cast, overload

import pandas as pd

from core.types import EnergySource, MetricType
from database.schemas.entsoe import EnergyGenerationSchema, EnergyLoadSchema


class EntsoeAdapter:
    _GENERATION_MAP = {
        "Biomass": EnergySource.BIOMASS,
        "Energy storage": EnergySource.BATTERY_STORAGE,
        "Fossil Brown coal/Lignite": EnergySource.FOSSIL_BROWN_COAL,
        "Fossil Gas": EnergySource.FOSSIL_GAS,
        "Fossil Hard coal": EnergySource.FOSSIL_HARD_COAL,
        "Fossil Oil": EnergySource.FOSSIL_OIL,
        "Fossil Oil shale": EnergySource.FOSSIL_OIL_SHALE,
        "Fossil Peat": EnergySource.FOSSIL_PEAT,
        "Fossil Coal-derived gas": EnergySource.FOSSIL_COAL_DERIVED_GAS,
        "Geothermal": EnergySource.GEOTHERMAL,
        "Hydro Pumped Storage": EnergySource.HYDRO_PUMPED_STORAGE,
        "Hydro Run-of-river and poundage": EnergySource.HYDRO_RUN_OF_RIVER,
        "Hydro Water Reservoir": EnergySource.HYDRO_WATER_RESERVOIR,
        "Marine": EnergySource.OTHER_RENEWABLE,
        "Nuclear": EnergySource.NUCLEAR,
        "Other": EnergySource.OTHER,
        "Other renewable": EnergySource.OTHER_RENEWABLE,
        "Solar": EnergySource.SOLAR,
        "Waste": EnergySource.WASTE,
        "Wind Onshore": EnergySource.WIND_ONSHORE,
        "Wind Offshore": EnergySource.WIND_OFFSHORE,
    }

    @overload
    def transform(
        self,
        df: pd.DataFrame,
        country_code: str,
        metric_type: Literal[MetricType.GENERATION],
    ) -> list[EnergyGenerationSchema]: ...

    @overload
    def transform(
        self, df: pd.DataFrame, country_code: str, metric_type: Literal[MetricType.LOAD]
    ) -> list[EnergyLoadSchema]: ...

    def transform(
        self, df: pd.DataFrame, country_code: str, metric_type: MetricType
    ) -> list[EnergyGenerationSchema] | list[EnergyLoadSchema]:
        """
        Transforms and validates ENTSO-E DataFrames into strict Pydantic models.
        """
        if df is None or df.empty:
            return []

        print(df.size)

        match metric_type:
            case MetricType.GENERATION:
                raw_dicts = self._transform_generation(df, country_code)
                return [EnergyGenerationSchema(**d) for d in raw_dicts]
            case MetricType.LOAD:
                raw_dicts = self._transform_load(df, country_code)
                return [EnergyLoadSchema(**d) for d in raw_dicts]
            case _:
                raise ValueError(f"Unsupported metric type: {metric_type}")

    def _transform_generation(
        self, df: pd.DataFrame, country_code: str
    ) -> list[dict[str, Any]]:
        df = self._prepare_index(df)
        flat_df = df.reset_index().melt(
            id_vars=[df.index.name or "index"],
            var_name="raw_energy_type",
            value_name="generation_mw",
        )
        flat_df.columns = pd.Index(["timestamp", "raw_energy_type", "generation_mw"])
        flat_df["energy_source"] = (
            flat_df["raw_energy_type"]
            .map(self._GENERATION_MAP)
            .fillna(EnergySource.OTHER)
        )
        flat_df = flat_df.drop(columns=["raw_energy_type"])
        flat_df["country_code"] = country_code

        return cast(
            list[dict[str, Any]],
            flat_df.dropna(subset=["generation_mw"]).to_dict(orient="records"),
        )

    def _transform_load(
        self, df: pd.DataFrame, country_code: str
    ) -> list[dict[str, Any]]:
        df = self._prepare_index(df)
        clean_df = df[["Actual Load"]].dropna().reset_index()
        clean_df.columns = pd.Index(["timestamp", "load_mw"])
        clean_df["country_code"] = country_code

        return cast(list[dict[str, Any]], clean_df.to_dict(orient="records"))

    def _prepare_index(self, df: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")
        return df
