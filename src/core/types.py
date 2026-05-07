from enum import Enum


class CountryCode(str, Enum):
    ROMANIA = "RO"
    HUNGARY = "HU"


class MetricType(str, Enum):
    """Metric Types"""

    GENERATION = "generation"
    LOAD = "load"


class EnergyDataProvider(str, Enum):
    """Data provider for the energy data."""

    ENTSOE = "entsoe"
    TRANSELECTRICA = "transelectrica"


class EnergySource(str, Enum):
    """Generation Sources"""

    # Renewables
    SOLAR = "solar"
    WIND_ONSHORE = "wind_onshore"
    WIND_OFFSHORE = "wind_offshore"
    HYDRO_WATER_RESERVOIR = "hydro_water_reservoir"
    HYDRO_RUN_OF_RIVER = "hydro_run_of_river"
    BIOMASS = "biomass"
    GEOTHERMAL = "geothermal"
    OTHER_RENEWABLE = "other_renewable"

    # Fossil/Thermal
    FOSSIL_GAS = "fossil_gas"
    FOSSIL_BROWN_COAL = "fossil_brown_coal_lignite"
    FOSSIL_HARD_COAL = "fossil_hard_coal"
    FOSSIL_OIL = "fossil_oil"
    FOSSIL_OIL_SHALE = "fossil_oil_shale"
    FOSSIL_COAL_DERIVED_GAS = "fossil_coal_derived_gas"
    FOSSIL_PEAT = "fossil_peat"

    # Storage/Other
    NUCLEAR = "nuclear"
    BATTERY_STORAGE = "battery_storage"
    HYDRO_PUMPED_STORAGE = "hydro_pumped_storage"
    WASTE = "waste"
    OTHER = "other"


class WeatherAreas(dict[str, float], Enum):
    BUCHAREST = {
        "latitude": 44.4323,
        "longitude": 26.1063,
    }
