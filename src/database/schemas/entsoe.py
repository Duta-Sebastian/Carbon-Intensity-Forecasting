from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, computed_field

from core.types import CountryCode, EnergyDataProvider, EnergySource


class EntsoeBaseSchema(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, from_attributes=True)

    timestamp: datetime = Field(..., description="UTC timestamp of the measurement")
    country_code: CountryCode = Field(
        ..., description="Country code for the measurement"
    )

    @computed_field
    @property
    def provider(self) -> EnergyDataProvider:
        return EnergyDataProvider.ENTSOE


class EnergyGenerationSchema(EntsoeBaseSchema):
    energy_source: EnergySource = Field(
        ..., description="e.g., 'load', 'solar', 'wind'"
    )
    generation_mw: float = Field(..., description="Generation value in MW")


class EnergyLoadSchema(EntsoeBaseSchema):
    load_mw: float = Field(..., description="Load value in MW")
