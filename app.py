"""
STEP 2 - The dashboard.   Run with:   streamlit run app.py
It loads the model trained by train_model.py and predicts the olive VARIETY
from the chemical composition typed by the user.
"""
import json, joblib, numpy as np, pandas as pd, altair as alt, streamlit as st

st.set_page_config(page_title="Olive Oil Variety Prediction", page_icon="🫒", layout="centered")

@st.cache_resource
def load():
    return joblib.load("model.joblib"), json.load(open("model_meta.json"))
model, meta = load()
F = meta["features"]; med = meta["medians"]; lo = meta["min"]; hi = meta["max"]

st.title("🫒 Predicting the Variety of Tunisian Olive Oils")
st.markdown("Enter the chemical composition of your olive oil sample. A Random Forest model "
            "trained on published data predicts its **variety**.")

# ---------------- 1. fatty acids ----------------
st.header("1. Fatty Acid Composition (% of total)")
c1, c2, c3 = st.columns(3)
vals = {}
def fa(col, key, label, step=0.1):
    vals[key] = col.number_input(label, 0.0, 100.0, float(round(med[key], 2)), step=step, format="%.2f")
fa(c1, "c160", "C16:0 (palmitic)"); fa(c2, "c180", "C18:0 (stearic)"); fa(c3, "c182", "C18:2 (linoleic)")
fa(c1, "c161", "C16:1 (palmitoleic)", 0.05); fa(c2, "c181", "C18:1 (oleic)"); fa(c3, "c183", "C18:3 (linolenic)", 0.05)

total = sum(vals[k] for k in ["c160","c161","c180","c181","c182","c183"])
st.caption(f"Sum of the 6 fatty acids: **{total:.1f} %** (usually 95–100 %; the rest are minor fatty acids)")
if total > 100:
    st.warning("The sum is above 100 % – please check your values.")

# ---------------- 2. quality indices ----------------
st.header("2. Quality Indices (optional)")
st.caption("Untick a box if you do not have the value: the model will use a typical value instead.")
q = [("acidite", "Acidity (%)", 0.05), ("peroxyde", "Peroxide (meq O₂/kg)", 0.5),
     ("k232", "K232", 0.05), ("k270", "K270", 0.01)]
cols = st.columns(4)
for col, (key, label, step) in zip(cols, q):
    with col:
        known = st.checkbox("known", value=(key in ["acidite", "peroxyde"]), key="k_" + key)
        v = st.number_input(label, 0.0, 100.0, float(round(med[key], 3)), step=step,
                            format="%.3f", disabled=not known, key="v_" + key)
        vals[key] = v if known else np.nan

st.markdown("---")

# ---------------- 3. prediction ----------------
if st.button("🔮 Predict", type="primary", width="stretch"):
    x = pd.DataFrame([vals])[F]
    proba = model.predict_proba(x)[0]
    order = np.argsort(proba)[::-1]
    best, conf = model.classes_[order[0]], proba[order[0]]

    st.header("Prediction Results")
    r1, r2 = st.columns(2)
    r1.metric("Predicted variety", best)
    r2.metric("Confidence", f"{conf:.0%}")
    if best == "Autres":
        st.info("“Autres” groups 11 rare varieties (Arbequina, Koroneiki, Coratina, Jarboui, Zalmati, "
                "Dhokar…) with too few samples to be predicted individually.")
    if conf < 0.5:
        st.warning("Low confidence: the profile is close to several varieties. Interpret with caution.")

    out = [k for k in F if not np.isnan(vals[k]) and not (lo[k] <= vals[k] <= hi[k])]
    if out:
        st.warning("Values outside the range of the training data (extrapolation, less reliable): "
                   + ", ".join(out))

    pdf = pd.DataFrame({"Variety": model.classes_, "Probability": proba})
    chart = (alt.Chart(pdf).mark_bar(color="#ff7a2f")
             .encode(x=alt.X("Probability:Q", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])),
                     y=alt.Y("Variety:N", sort="-x"), tooltip=["Variety", alt.Tooltip("Probability:Q", format=".0%")])
             .properties(height=200))
    st.altair_chart(chart, width="stretch")

    st.subheader("Your sample vs. the average profile of each variety")
    prof = pd.DataFrame(meta["profiles"]).T[F]
    prof.loc["▶ Your sample"] = [vals[k] for k in F]
    st.dataframe(prof.round(2), width="stretch")

with st.expander("ℹ️ About the model"):
    cv = meta["cv"]
    st.write(f"Random Forest (500 trees) trained on **{meta['n_samples']} samples** from published studies "
             f"of Tunisian olive oils. Classes: {', '.join(meta['classes'])}.")
    st.write(f"Estimated performance (cross-validation, samples of the same publication kept apart): "
             f"accuracy **{cv['accuracy']:.0%}**, balanced accuracy **{cv['balanced_accuracy']:.0%}**.")
    st.write("Most influential variables:")
    imp = pd.Series(meta["importance"]).sort_values(ascending=False)
    st.bar_chart(imp)
    st.caption("Small dataset: this is a proof of concept, not a certified authentication tool.")
