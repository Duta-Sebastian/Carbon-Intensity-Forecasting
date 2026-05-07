from sqlalchemy import Column, DateTime, Float, Index
from sqlalchemy import Enum as SQLEnum

from core.types import CountryCode, EnergyDataProvider, EnergySource
from database.models.Base import Base


class EnergyGeneration(Base):
    __tablename__ = "energy_generation"

    timestamp = Column(DateTime(timezone=True), primary_key=True, index=True)
    country_code = Column(SQLEnum(CountryCode), primary_key=True)
    provider = Column(SQLEnum(EnergyDataProvider), primary_key=True)
    energy_source = Column(SQLEnum(EnergySource), primary_key=True)

    updated_at = Column(DateTime(timezone=True))
    generation_mw = Column(Float, nullable=False)

    __table_args__ = (
        Index(
            "idx_provider_source_time_country",
            "provider",
            "energy_source",
            "country_code",
            timestamp.desc(),
            unique=True,
        ),
    )

    # op.execute("SELECT create_hypertable('energy_generation', 'timestamp')")
