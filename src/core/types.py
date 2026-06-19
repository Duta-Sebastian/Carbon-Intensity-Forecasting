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


class WeatherDataProvider(str, Enum):
    """Data provider for weather data"""

    OPENMETEO = "openmeteo"


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


class RomanianCity(Enum):
    BUCHAREST = ("Bucharest", 44.4268, 26.1025)
    CLUJ_NAPOCA = ("Cluj-Napoca", 46.7712, 23.6236)
    TIMISOARA = ("Timișoara", 45.7489, 21.2087)
    IASI = ("Iași", 47.1585, 27.6014)
    CONSTANTA = ("Constanța", 44.1598, 28.6348)
    BRASOV = ("Brașov", 45.6427, 25.5887)
    GALATI = ("Galați", 45.4353, 28.0075)
    PLOIESTI = ("Ploiești", 44.9407, 26.0296)
    ORADEA = ("Oradea", 47.0465, 21.9189)
    BACAU = ("Bacău", 46.5673, 26.9136)

    @property
    def city(self):
        return self.value[0]

    @property
    def lat(self):
        return self.value[1]

    @property
    def lon(self):
        return self.value[2]


CITY_POPULATION_MAP: dict[RomanianCity, int] = {
    RomanianCity.BUCHAREST: 1716961,
    RomanianCity.CLUJ_NAPOCA: 286598,
    RomanianCity.TIMISOARA: 250849,
    RomanianCity.IASI: 271692,
    RomanianCity.CONSTANTA: 263688,
    RomanianCity.BRASOV: 237589,
    RomanianCity.GALATI: 217851,
    RomanianCity.PLOIESTI: 180540,
    RomanianCity.ORADEA: 183105,
    RomanianCity.BACAU: 136087,
}
