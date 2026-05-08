from sqlalchemy import Column, DateTime, Float, Index
from sqlalchemy import Enum as SQLEnum

from core.types import CountryCode, EnergyDataProvider
from database.models.Base import Base


class EnergyLoad(Base):
    __tablename__ = "energy_load"

    timestamp = Column(DateTime(timezone=True), primary_key=True, index=True)
    country_code = Column(SQLEnum(CountryCode), primary_key=True)
    provider = Column(SQLEnum(EnergyDataProvider), primary_key=True)

    updated_at = Column(DateTime(timezone=True))
    load_mw = Column(Float, nullable=False)

    __table_args__ = (
        Index(
            "idx_energy_load_provider_time_country",
            "provider",
            "country_code",
            timestamp.desc(),
            unique=True,
        ),
    )

    # op.execute("SELECT create_hypertable('energy_load', 'timestamp')")
