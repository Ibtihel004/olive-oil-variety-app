import pandas as pd, numpy as np, re

SRC = "4_5857419313874477745.xlsx"   # <- the raw Excel file, in the SAME folder as this script
df = pd.read_excel(SRC)
n0 = len(df)

# 1. strip tabs / spaces in all text columns
for c in df.select_dtypes(include=["object", "string"]).columns:
    df[c] = df[c].astype("string").str.replace(r"\\t|\t", " ", regex=True).str.strip().str.replace(r"\s+", " ", regex=True)

# 2. "_" -> NaN, numeric columns -> float
df = df.replace({"_": pd.NA})
num = ["c160","c161","c180","c181","c182","c183","acidite","peroxyde","k232","k270","deltak"]
for c in num:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# 3. harmonise variety names (original kept for traceability)
df["variete_originale"] = df["variete"]
vmap = {"Chétoui":"Chetoui","Chemlali Zarzis":"Chemlali","Chemlali Boughrara":"Chemlali",
        "Chemlali Sfax":"Chemlali","Arbequina I-18":"Arbequina","Ain Jarboua":"Jarboui"}
df["variete"] = df["variete"].replace(vmap)

# 4. harmonise classeReelle
df["classeReelle"] = df["classeReelle"].str.strip()

# 5. dates (mixed formats: year only, ISO, mm/dd/yyyy)
def parse_date(x):
    s = str(x).strip()
    if re.fullmatch(r"\d{4}", s): return pd.Timestamp(int(s), 1, 1)
    return pd.to_datetime(s, errors="coerce")
df["dateEch"] = df["dateEch"].apply(parse_date)
df["anneeEch"] = df["dateEch"].dt.year

# 6. origin -> region (simple, from text)
def region(o):
    o = o.lower()
    if any(k in o for k in ["beja","northern","thibar","tibar","amdoun","kef","nebeur"]): return "Nord"
    if any(k in o for k in ["sfax"]): return "Centre-Est"
    return "Sud"
df["region"] = df["origine"].apply(region)

# 7. duplicates
dups = df.duplicated(subset=num + ["variete"]).sum()
df = df.drop_duplicates(subset=num + ["variete"])

# 8. columns not usable as features / leakage / useless
#   classAuto & alerte are computed FROM the chemical values -> redundant; sourceDonnees constant; lien = text
df_model = df.drop(columns=["classAuto","alerte","sourceDonnees","lienPublication","variete_originale","dateEch","origine","idEch"])
# deltak: >70% missing -> drop
miss = df[num].isna().mean().round(3)
df_model = df_model.drop(columns=["deltak"])

# 9. flag suspicious values (kept, not deleted)
flags = []
for i, r in df.iterrows():
    f = []
    if pd.notna(r.acidite) and r.acidite > 2: f.append("acidite>2")
    if pd.notna(r.peroxyde) and r.peroxyde > 20: f.append("peroxyde>20")
    if pd.notna(r.k232) and r.k232 > 2.6: f.append("k232>2.6")
    if pd.notna(r.k270) and r.k270 > 0.25: f.append("k270>0.25")
    flags.append(", ".join(f))
df["valeurs_suspectes"] = flags
df_model["valeurs_suspectes"] = df["valeurs_suspectes"]

vc = df["variete"].value_counts().rename_axis("variete").reset_index(name="n")
with pd.ExcelWriter("olive_oil_clean.xlsx") as w:
    df.to_excel(w, sheet_name="donnees_nettoyees", index=False)
    df_model.to_excel(w, sheet_name="pour_modele", index=False)
    vc.to_excel(w, sheet_name="varietes", index=False)
    miss.rename("pct_manquant").reset_index().to_excel(w, sheet_name="valeurs_manquantes", index=False)
df.to_csv("olive_oil_clean.csv", index=False)

print("rows", n0, "->", len(df), "dups removed", dups)
print(vc.to_string())
print(miss)
print(df[df.valeurs_suspectes!=""][["idEch","variete","acidite","peroxyde","k232","k270","valeurs_suspectes"]])
print(df.groupby("region").size())
print(df_model.columns.tolist())
