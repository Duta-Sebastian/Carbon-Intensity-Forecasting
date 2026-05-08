import pandas as pd
from geopy.extra.rate_limiter import RateLimiter
from geopy.geocoders import Nominatim

df = pd.read_excel(
    "data/Wind_Stations_Analysis.xlsx",
    sheet_name="Geographic_Nodes_Manual",
    usecols=["Județul", "Stația de racord (UAT)", "Putere Totală (MW)"],
    skipfooter=1,
)

df["Full Address"] = "Romania, " + df["Județul"] + ", " + df["Stația de racord (UAT)"]

geolocator = Nominatim(user_agent="romania_energy_mapper")
geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)

df["location"] = df["Full Address"].apply(geocode)

df["latitude"] = df["location"].apply(lambda loc: loc.latitude if loc else None)
df["longitude"] = df["location"].apply(lambda loc: loc.longitude if loc else None)

df_final = df.drop(columns=["location"])
df_final.to_csv("data/raw_geocoded_backup.csv", index=False)

print(df)
