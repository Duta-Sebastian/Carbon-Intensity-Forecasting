import re

import pandas as pd


def clean_station_name(text):
    if not isinstance(text, str):
        return ""

    # 1. Trecem totul în litere mari pentru a uniformiza
    text = text.upper()

    # 2. Eliminăm diacriticele de bază (staţia -> statia)
    text = (
        text.replace("Ţ", "T")
        .replace("Ş", "S")
        .replace("Ă", "A")
        .replace("Â", "A")
        .replace("Î", "I")
    )

    # 3. Eliminăm tensiunile (ex: 110/20/6KV, 110 KV, 20/0,4)
    # Caută cifre despărțite de /, , sau . urmate opțional de K, V, KV
    text = re.sub(r"\d+(?:[.,/]\d+)*\s*K?V\b", "", text)

    # 4. Eliminăm cuvinte tehnice și abrevieri specifice
    tech_words = [
        r"\bSTATIA\b",
        r"\bLEA\b",
        r"\bPTA\b",
        r"\bPTAB\b",
        r"\bPT\b",
        r"\bCT\b",
        r"\bSN\b",
        r"\bJT\b",
        r"\bDIN\b",
        r"\bA\d+\b",
    ]
    for word in tech_words:
        text = re.sub(word, "", text)

    # 5. Tratăm liniile de legătură (ex: "Basarabi - Cobadin" sau "Oravita; LEA...")
    # Păstrăm doar prima parte (până la primul cratimă sau punct și virgulă)
    text = text.split("-")[0]
    text = text.split(";")[0]

    # 6. Eliminăm numerele rămase izolate (ex: PTA 406 -> rămâne 406 -> ștergem)
    text = re.sub(r"\b\d+[A-Z]?\b", "", text)

    # 7. Curățăm caracterele speciale rămase și spațiile multiple
    text = re.sub(r"[^A-Z\s]", " ", text)
    text = " ".join(text.split())  # Elimină spațiile duble

    return text


# df = pd.read_excel("data/GCAs.xlsx", sheet_name="Eolian", skiprows=13)  # noqa: F821

# # 2. Realiniere (pentru rândurile cu "EOLIAN" la Județ)
# mask_decalat = df["Județul"].astype(str).str.contains("EOLIAN", case=False, na=False)

# if mask_decalat.any():
#     df = df.astype(object)
#     all_columns = df.columns.tolist()
#     idx_judet = all_columns.index("Județul")
#     idx_nr_crt_2 = all_columns.index("Nr. CR")

#     for i in df[mask_decalat].index:
#         values_to_shift = df.iloc[i, idx_judet + 1 : idx_nr_crt_2].values
#         df.iloc[i, idx_judet : idx_nr_crt_2 - 1] = values_to_shift

# # 3. Curățare (eliminăm rândurile unde "Denumire investitor" este gol)
# # Folosim dropna pe coloana de investitor.
# # Verifică dacă în Excel-ul tău coloana se numește exact așa (fără 's' la final)
# df = df.dropna(subset=["Denumire investitor"])

# # 4. Eliminăm coloana "Nr. crt." (sau "Nr.crt.")
# # Folosim o listă de posibile nume pentru a fi siguri că o găsim
# cols_to_drop = [c for c in df.columns if "Nr." in c or "crt" in c.lower()]
# df = df.drop(columns=cols_to_drop)

# # 5. Salvare Finală
# # index=False face ca Pandas să NU scrie coloana de numere 0, 1, 2... în stânga
# output_path = "data/GCAs_Clean.xlsx"
# df.to_excel(output_path, index=False)

# print(f"Succes! Fișierul a fost salvat la: {output_path}")
# if mask_decalat.any():
#     print("\nExemplu de rând corectat:")
#     print(
#         df[mask_decalat][
#             [
#                 "Denumire investitor",
#                 "Județul",
#                 "Staţia de racord",
#                 "Putere PIF conform emitenți (MW)",
#             ]
#         ].head()
#     )

used_columns = [
    "Denumire investitor",
    "Denumire centrale electrice eoliene",
    "Județul",
    "Staţia de racord",
    "Putere PIF conform emitenți (MW)",
]

gca_wind = pd.read_excel("data/GCAs_Clean.xlsx", usecols=used_columns)
print(len(gca_wind))
gca_wind_clean = gca_wind.dropna(
    subset=[
        "Denumire investitor",
        "Județul",
        "Staţia de racord",
        "Putere PIF conform emitenți (MW)",
    ]
)
print(len(gca_wind_clean))
gca_wind_clean["Putere PIF conform emitenți (MW)"] = pd.to_numeric(
    gca_wind_clean["Putere PIF conform emitenți (MW)"], errors="coerce"
)

gca_wind_clean = gca_wind_clean.dropna(subset=["Putere PIF conform emitenți (MW)"])

gca_wind_clean = gca_wind_clean[gca_wind_clean["Putere PIF conform emitenți (MW)"] > 0]

gca_wind_clean.to_excel("data/GCAs_Final.xlsx", index=False)

print(f"Rânduri rămase după curățare: {len(gca_wind_clean)}")
total_pif = gca_wind_clean["Putere PIF conform emitenți (MW)"].sum()

print(f"Capacitatea totală instalată (PIF): {total_pif:.2f} MW")
print(gca_wind_clean.head())

gca_wind_clean["Locatie_Curata"] = gca_wind_clean["Staţia de racord"].apply(
    clean_station_name
)

# --- PASUL 3: Agregarea Granulară (Stații de racord individuale) ---
# Asta e lista ta originală, utilă pentru audit tehnic
unique_stations_granular = (
    gca_wind_clean.groupby(["Județul", "Staţia de racord"])[
        "Putere PIF conform emitenți (MW)"
    ]
    .sum()
    .reset_index()
)

# --- PASUL 4: Agregarea Geografică (Locații curățate) ---
# Asta e lista pentru Geocoding, unde "110kV Baia" și "20kV Baia" devin un singur punct "BAIA"
unique_stations_aggregated = (
    gca_wind_clean.groupby(["Județul", "Locatie_Curata"])[
        "Putere PIF conform emitenți (MW)"
    ]
    .sum()
    .reset_index()
)

with pd.ExcelWriter("data/Wind_Stations_Analysis.xlsx") as writer:
    unique_stations_granular.to_excel(writer, sheet_name="Granular_Tech", index=False)
    unique_stations_aggregated.to_excel(
        writer, sheet_name="Geographic_Nodes", index=False
    )

print(f"S-au salvat {len(unique_stations_granular)} intrări granulare.")
print(f"S-au salvat {len(unique_stations_aggregated)} noduri geografice unice.")
