from pathlib import Path
import io
import joblib
import librosa
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# AURA NEXUS — uses the team's supplied Random Forest and exact seven-feature pipeline.
st.set_page_config(page_title="AURA NEXUS | Acoustic Fault Intelligence", page_icon="◈", layout="wide")

ROOT = Path(__file__).resolve().parent
MODEL_FILE = ROOT / "AURA_RandomForest_Model.pkl"
SCHEMA_FILE = ROOT / "AURA_Feature_Schema.pkl"
DATA_FILE = ROOT / "AURA_Feature_Dataset.xlsx"
NORMAL_FILE = ROOT / "DEMO_Normal.wav"
ANOMALY_FILE = ROOT / "DEMO_Anomaly.wav"

FEATURES = [
    "RMS_mean", "RMS_std", "Peak_amplitude", "Dominant_frequency",
    "Spectral_centroid", "Spectral_bandwidth", "Spectral_rolloff"
]
FEATURE_LABELS = {
    "RMS_mean": "Mean RMS energy",
    "RMS_std": "RMS variation",
    "Peak_amplitude": "Peak amplitude",
    "Dominant_frequency": "Dominant frequency",
    "Spectral_centroid": "Spectral centroid",
    "Spectral_bandwidth": "Spectral bandwidth",
    "Spectral_rolloff": "Spectral roll-off"
}
FEATURE_PHYSICS = {
    "RMS_mean": "Represents the average short-time signal energy. Multiplying the waveform by a constant scales RMS by the same factor.",
    "RMS_std": "Describes how much short-time signal energy fluctuates during the recording. Constant amplitude scaling also scales this value.",
    "Peak_amplitude": "The largest absolute sample value. Constant amplitude scaling changes the peak in direct proportion.",
    "Dominant_frequency": "The frequency bin with the strongest non-zero FFT magnitude. Ideal constant amplitude scaling does not move its frequency location.",
    "Spectral_centroid": "The magnitude-weighted centre of the spectrum. Ideal constant amplitude scaling changes magnitude but not the centroid location.",
    "Spectral_bandwidth": "The spread of spectral energy around the centroid. Ideal constant amplitude scaling does not change the frequency spread.",
    "Spectral_rolloff": "A frequency boundary containing most of the spectral energy. Ideal constant amplitude scaling does not move this boundary."
}

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');
:root{--bg:#070b13;--panel:#101a29;--line:rgba(135,171,211,.18);--cyan:#59e7f2;--mint:#9affd5;--violet:#9b9cff;--pink:#ff819d;--muted:#94a9c0}
html,body,[class*="css"]{font-family:'Manrope',sans-serif}
.stApp{background:radial-gradient(ellipse at 78% 0%,rgba(42,105,160,.22),transparent 34%),radial-gradient(ellipse at 0% 45%,rgba(34,190,180,.10),transparent 30%),var(--bg);color:#edf5ff}
.block-container{max-width:1540px;padding-top:1.3rem;padding-bottom:3rem}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#0d1726,#090f19);border-right:1px solid var(--line)}
[data-testid="stSidebar"] *{color:#e8f3ff}
.hero{border:1px solid rgba(89,231,242,.28);border-radius:25px;padding:28px 32px;background:linear-gradient(112deg,rgba(19,39,61,.98),rgba(12,21,36,.94) 62%,rgba(30,43,76,.94));box-shadow:0 18px 60px rgba(0,0,0,.24)}
.eyebrow{font: .7rem 'DM Mono',monospace;letter-spacing:.2em;color:var(--cyan);text-transform:uppercase}
.hero h1{font-size:2.45rem;line-height:1.08;letter-spacing:-.05em;margin:.55rem 0 .45rem;color:#f5f9ff}
.hero p{color:#a9bdd2;margin:0;font-size:1rem}
.hero-chip{display:inline-block;border:1px solid rgba(89,231,242,.25);background:rgba(89,231,242,.08);border-radius:99px;padding:6px 11px;color:#baf8ff;font: .68rem 'DM Mono',monospace;margin-right:6px;margin-top:15px}
.section-kicker{font: .68rem 'DM Mono',monospace;color:var(--cyan);letter-spacing:.16em;text-transform:uppercase}
.panel{border:1px solid var(--line);border-radius:17px;background:linear-gradient(145deg,rgba(17,30,48,.96),rgba(11,20,33,.96));padding:17px 19px}
.note{border:1px solid rgba(89,231,242,.22);border-radius:13px;background:rgba(32,73,93,.18);padding:13px 16px;color:#c3d6e8}
.stButton>button,.stDownloadButton>button{border:1px solid rgba(89,231,242,.4);background:linear-gradient(90deg,rgba(89,231,242,.16),rgba(155,156,255,.16));color:#f0fcff;border-radius:12px;min-height:2.6rem}
.stButton>button:hover{border-color:var(--cyan);color:white;box-shadow:0 0 18px rgba(89,231,242,.15)}
[data-testid="stMetric"]{background:linear-gradient(145deg,rgba(20,35,54,.96),rgba(12,21,34,.96));border:1px solid var(--line);padding:15px 17px;border-radius:16px}
[data-testid="stMetricLabel"]{color:#9db2c9}
[data-testid="stFileUploader"]{background:#111d2d;border:1px dashed rgba(89,231,242,.42);border-radius:15px}
.stTabs [data-baseweb="tab-list"]{gap:7px}
.stTabs [data-baseweb="tab"]{background:#101a29;border:1px solid var(--line);border-radius:10px 10px 0 0;padding:10px 16px}
.stTabs [aria-selected="true"]{color:var(--cyan);border-color:rgba(89,231,242,.55)}
div[data-testid="stExpander"]{border:1px solid var(--line);border-radius:14px;background:rgba(14,24,39,.72)}
hr{border-color:var(--line)}
footer{visibility:hidden}
.small{color:#91a7bf;font-size:.82rem}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_FILE)
    schema = list(joblib.load(SCHEMA_FILE))
    if schema != FEATURES:
        raise ValueError(f"Feature order mismatch: {schema}")
    if getattr(model, "n_features_in_", None) != len(schema):
        raise ValueError("The model input size does not match the seven-feature schema.")
    return model, schema

@st.cache_data
def load_feature_table():
    if not DATA_FILE.exists():
        return None
    data = pd.read_excel(DATA_FILE)
    required = FEATURES + ["Label"]
    if not set(required).issubset(data.columns):
        return None
    return data

def load_audio_bytes(raw):
    y, sr = librosa.load(io.BytesIO(raw), sr=None, mono=True)
    if y.size == 0 or not np.all(np.isfinite(y)):
        raise ValueError("Audio file is empty or invalid.")
    return y.astype(np.float32), int(sr)

def extract_features(y, sr):
    # Keep this identical to the team's training notebook feature definitions.
    y = np.asarray(y, dtype=np.float32)
    rms_values = librosa.feature.rms(y=y)[0]
    fft_values = np.abs(np.fft.rfft(y))
    frequencies = np.fft.rfftfreq(len(y), d=1 / sr)
    dominant_frequency = float(frequencies[np.argmax(fft_values[1:]) + 1]) if len(fft_values) > 1 else 0.0
    return {
        "RMS_mean": float(np.mean(rms_values)),
        "RMS_std": float(np.std(rms_values)),
        "Peak_amplitude": float(np.max(np.abs(y))),
        "Dominant_frequency": dominant_frequency,
        "Spectral_centroid": float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))),
        "Spectral_bandwidth": float(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))),
        "Spectral_rolloff": float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)))
    }

def predict(model, schema, feat):
    X = pd.DataFrame([[feat[k] for k in schema]], columns=schema)
    pred = int(model.predict(X)[0])
    probas = model.predict_proba(X)[0]
    classes = list(model.classes_)
    p_anomaly = float(probas[classes.index(1)]) if 1 in classes else float("nan")
    return pred, p_anomaly

def chart_style(fig, title, x_title, y_title, height=330):
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color="#edf6ff")),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Manrope", color="#9eb3ca", size=11),
        xaxis_title=x_title, yaxis_title=y_title, height=height,
        margin=dict(l=12,r=12,t=48,b=12),
        xaxis=dict(gridcolor="rgba(130,160,190,.12)"),
        yaxis=dict(gridcolor="rgba(130,160,190,.12)"),
        showlegend=False
    )
    return fig

def waveform_chart(y, sr, start=0.0, duration=None):
    duration = len(y)/sr if duration is None else duration
    lo = max(0, int(start*sr)); hi = min(len(y), max(lo+1, int((start+duration)*sr)))
    part = y[lo:hi]
    step = max(1, len(part)//9000)
    t = (np.arange(lo, hi, step))/sr
    fig = go.Figure(go.Scatter(x=t, y=part[::step], mode="lines",
        line=dict(color="#59e7f2",width=1.15),fill="tozeroy",fillcolor="rgba(89,231,242,.08)"))
    return chart_style(fig, "01 / TIME-DOMAIN WAVEFORM", "Time (s)", "Amplitude", 300)

def spectrum_chart(y, sr, max_hz):
    mag = np.abs(np.fft.rfft(y))
    freq = np.fft.rfftfreq(len(y), d=1/sr)
    keep = freq <= max_hz
    fig = go.Figure(go.Scatter(x=freq[keep], y=mag[keep], mode="lines",
        line=dict(color="#9affd5",width=1.2),fill="tozeroy",fillcolor="rgba(154,255,213,.06)"))
    return chart_style(fig, "02 / FFT FREQUENCY SIGNATURE", "Frequency (Hz)", "Magnitude", 300)

def probability_curve(y, sr, model, schema):
    factors = np.round(np.arange(0.25, 2.001, 0.05), 2)
    rows=[]
    for factor in factors:
        f = extract_features(y*factor, sr)
        cls, p = predict(model, schema, f)
        rows.append({"Amplitude multiplier":factor, "Anomaly probability (%)":p*100,
                     "Predicted class":"Anomalous" if cls==1 else "Normal",
                     "RMS mean":f["RMS_mean"], "RMS std":f["RMS_std"],
                     "Peak amplitude":f["Peak_amplitude"]})
    return pd.DataFrame(rows)

def fmt_class(pred):
    return "ANOMALOUS" if pred==1 else "NORMAL"

try:
    if not MODEL_FILE.exists() or not SCHEMA_FILE.exists():
        st.error("Model artifacts are missing. Keep the two AURA .pkl files beside this app.")
        st.stop()
    model, schema = load_artifacts()
except Exception as exc:
    st.error(f"Could not load the team's trained model: {exc}")
    st.stop()

# Sidebar input station
with st.sidebar:
    st.markdown('<div class="eyebrow">AURA NEXUS / INPUT STATION</div>', unsafe_allow_html=True)
    st.markdown("## Select machine signal")
    input_mode = st.radio("Audio source", ["Normal demo", "Anomalous demo", "Upload WAV"], index=0)
    uploaded = None
    if input_mode == "Upload WAV":
        uploaded = st.file_uploader("Upload machine recording", type=["wav"])
    st.markdown("---")
    st.markdown("**MODEL STATUS**")
    st.success("Random Forest • 7 trained features")
    st.markdown("**OUTPUT CLASSES**")
    st.write("0 — Normal")
    st.write("1 — Anomalous")
    st.markdown("---")
    st.caption("Prediction is a model classification, not a confirmed mechanical diagnosis.")

# Load selected audio
try:
    if input_mode == "Normal demo":
        if not NORMAL_FILE.exists():
            st.error("DEMO_Normal.wav is missing from the project folder."); st.stop()
        audio_bytes = NORMAL_FILE.read_bytes()
        signal, sr = load_audio_bytes(audio_bytes)
        signal_name = NORMAL_FILE.name
    elif input_mode == "Anomalous demo":
        if not ANOMALY_FILE.exists():
            st.error("DEMO_Anomaly.wav is missing from the project folder."); st.stop()
        audio_bytes = ANOMALY_FILE.read_bytes()
        signal, sr = load_audio_bytes(audio_bytes)
        signal_name = ANOMALY_FILE.name
    else:
        if uploaded is None:
            st.markdown('<div class="hero"><div class="eyebrow">INDUSTRIAL SOUND INTELLIGENCE / LIVE LAB</div><h1>AURA <span style="color:#59e7f2">NEXUS</span></h1><p>Upload a machine WAV recording, then explore its waveform, frequency signature and AI classification.</p><span class="hero-chip">RANDOM FOREST</span><span class="hero-chip">7 ACOUSTIC FEATURES</span><span class="hero-chip">PHYSICS-INFORMED SENSITIVITY</span></div>', unsafe_allow_html=True)
            st.info("Choose a built-in demo or upload a WAV recording from the left panel to start.")
            st.stop()
        audio_bytes = uploaded.getvalue()
        signal, sr = load_audio_bytes(audio_bytes)
        signal_name = uploaded.name
except Exception as exc:
    st.error(f"Audio could not be analyzed: {exc}")
    st.stop()

duration = len(signal)/sr
features = extract_features(signal, sr)
prediction, anomaly_probability = predict(model, schema, features)
label = fmt_class(prediction)
color = "#ff819d" if prediction==1 else "#9affd5"

# Hero
st.markdown(f"""
<div class="hero">
 <div class="eyebrow">INDUSTRIAL SOUND INTELLIGENCE / SESSION ACTIVE</div>
 <h1>AURA <span style="color:#59e7f2">NEXUS</span></h1>
 <p>Acoustic Fault Detection for Machines — translating machine sound into measurable features and an AI-supported condition prediction.</p>
 <span class="hero-chip">TEAM RANDOM FOREST MODEL</span><span class="hero-chip">NORMAL / ANOMALOUS</span><span class="hero-chip">SIGNAL → FEATURES → PREDICTION</span>
</div>
""", unsafe_allow_html=True)
st.write("")
st.markdown(f'<div class="small">ACTIVE RECORDING: <b>{signal_name}</b> &nbsp; • &nbsp; {duration:.2f} s &nbsp; • &nbsp; {sr:,} Hz &nbsp; • &nbsp; {len(signal):,} samples</div>', unsafe_allow_html=True)
st.audio(audio_bytes, format="audio/wav")

# Main readouts with compact labels to avoid clipping.
m1,m2,m3,m4 = st.columns(4)
m1.metric("AI-PREDICTED CONDITION", label)
m2.metric("ANOMALY PROBABILITY", f"{anomaly_probability*100:.1f}%")
m3.metric("RECORDING LENGTH", f"{duration:.2f} s")
m4.metric("SAMPLING RATE", f"{sr/1000:.1f} kHz")
st.write("")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "◈ Signal Explorer", "⇄ State Comparison", "⌁ Sensitivity Lab", "▦ Feature Intelligence", "◎ Model & Dataset"
])

with tab1:
    st.markdown("### Decode the acoustic signal")
    st.markdown('<div class="small">The waveform shows amplitude over time. The FFT spectrum shows how signal magnitude is distributed across frequency.</div>', unsafe_allow_html=True)
    c1,c2 = st.columns(2)
    with c1:
        start = st.slider("Waveform window start (s)", 0.0, max(0.0, float(duration-0.5)), 0.0, 0.1)
    with c2:
        max_window = max(0.5, min(float(duration), 10.0))
        window = st.slider("Waveform window length (s)", 0.5, max_window, min(5.0,max_window), 0.5)
    max_freq = st.slider("FFT display limit (Hz)", 500, max(500, int(sr/2)), min(8000,max(500,int(sr/2))), 500)
    left,right = st.columns(2)
    with left:
        st.plotly_chart(waveform_chart(signal,sr,start,window),use_container_width=True,config={"displayModeBar":True})
    with right:
        st.plotly_chart(spectrum_chart(signal,sr,max_freq),use_container_width=True,config={"displayModeBar":True})
    st.markdown('<div class="note"><b>How to read it:</b> waveform amplitude describes signal strength and its variation with time. FFT peaks identify strong frequency components. A changed spectrum can indicate a changed acoustic pattern, but this classifier outputs only Normal or Anomalous—not a specific fault type.</div>', unsafe_allow_html=True)

with tab2:
    st.markdown("### Compare the two reference machine states")
    st.markdown('<div class="small">Both reference WAV files are passed through the same trained model and the same feature extraction pipeline.</div>', unsafe_allow_html=True)
    if NORMAL_FILE.exists() and ANOMALY_FILE.exists():
        yn,srn = load_audio_bytes(NORMAL_FILE.read_bytes())
        ya,sra = load_audio_bytes(ANOMALY_FILE.read_bytes())
        fn,pn,prn = extract_features(yn,srn), None, None
        pn,prn = predict(model,schema,fn)
        fa = extract_features(ya,sra)
        pa,pra = predict(model,schema,fa)
        a,b = st.columns(2)
        with a:
            st.markdown("#### Reference: Normal")
            st.audio(NORMAL_FILE.read_bytes(),format="audio/wav")
            st.metric("Model prediction",fmt_class(pn))
            st.metric("Anomaly probability",f"{prn*100:.1f}%")
            st.plotly_chart(waveform_chart(yn,srn),use_container_width=True,config={"displayModeBar":False})
        with b:
            st.markdown("#### Reference: Anomalous")
            st.audio(ANOMALY_FILE.read_bytes(),format="audio/wav")
            st.metric("Model prediction",fmt_class(pa))
            st.metric("Anomaly probability",f"{pra*100:.1f}%")
            st.plotly_chart(waveform_chart(ya,sra),use_container_width=True,config={"displayModeBar":False})
        compare = pd.DataFrame([
            {"Reference recording":"Normal demo","Predicted class":fmt_class(pn),"Anomaly probability (%)":round(prn*100,2)},
            {"Reference recording":"Anomalous demo","Predicted class":fmt_class(pa),"Anomaly probability (%)":round(pra*100,2)}
        ])
        st.dataframe(compare,use_container_width=True,hide_index=True)
        st.caption("These are two demonstration samples, not a substitute for evaluation on a held-out test set.")
    else:
        st.warning("The two demo WAV files are required for this comparison.")

with tab3:
    st.markdown("### Controlled input–output sensitivity experiment")
    st.markdown('<div class="note"><b>Experiment question:</b> If the amplitude of the same recorded sound is multiplied by a controlled factor, how do amplitude-related acoustic features and the model output respond?</div>', unsafe_allow_html=True)
    factor = st.slider("Amplitude multiplier (×)",0.25,2.00,1.00,0.05,key="nexus_amp")
    modified = signal*factor
    modified_features = extract_features(modified,sr)
    modified_class, modified_prob = predict(model,schema,modified_features)
    q1,q2,q3,q4 = st.columns(4)
    q1.metric("Predicted condition",fmt_class(modified_class))
    q2.metric("Anomaly probability",f"{modified_prob*100:.1f}%",f"{(modified_prob-anomaly_probability)*100:+.1f} percentage points")
    q3.metric("Mean RMS",f"{modified_features['RMS_mean']:.5f}",f"{modified_features['RMS_mean']-features['RMS_mean']:+.5f}")
    q4.metric("Peak amplitude",f"{modified_features['Peak_amplitude']:.5f}",f"{modified_features['Peak_amplitude']-features['Peak_amplitude']:+.5f}")
    curve = probability_curve(signal,sr,model,schema)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=curve["Amplitude multiplier"],y=curve["Anomaly probability (%)"],mode="lines+markers",
        line=dict(color="#9b9cff",width=3),marker=dict(color="#59e7f2",size=5),
        fill="tozeroy",fillcolor="rgba(155,156,255,.10)",name="Anomaly probability"))
    fig.add_vline(x=factor,line_dash="dot",line_color="#59e7f2",annotation_text="Selected amplitude",annotation_font_color="#59e7f2")
    fig.update_yaxes(range=[0,100])
    st.plotly_chart(chart_style(fig,"MODEL RESPONSE CURVE / AMPLITUDE SWEEP","Amplitude multiplier (×)","Anomaly probability (%)",370),use_container_width=True)
    st.markdown("#### Physics interpretation")
    st.write(f"At an amplitude multiplier of **{factor:.2f}×**, the mean RMS changed from **{features['RMS_mean']:.5f}** to **{modified_features['RMS_mean']:.5f}**, and peak amplitude changed from **{features['Peak_amplitude']:.5f}** to **{modified_features['Peak_amplitude']:.5f}**.")
    st.write("For ideal constant scaling, the waveform's amplitude-related quantities (RMS and peak amplitude) scale with the multiplier. Frequency locations and normalized spectral shape descriptors are theoretically unchanged by a positive constant multiplier. The Random Forest may still change its class probability because it evaluates the complete seven-feature vector against patterns learned during training.")
    st.warning("Important limitation: digital amplitude scaling is a controlled signal-processing experiment. It does not create a real mechanical fault, and an amplitude change can also arise from microphone distance or recording gain. Interpret the model response as sensitivity to this modified input—not proof of a fault mechanism.")
    with st.expander("Inspect and download all sensitivity points"):
        st.dataframe(curve,use_container_width=True,hide_index=True)
        st.download_button("Download sensitivity sweep (CSV)",curve.to_csv(index=False).encode("utf-8"),"AURA_NEXUS_Sensitivity.csv","text/csv")

with tab4:
    st.markdown("### Seven-feature acoustic fingerprint")
    st.markdown("<div class=\"small\">The features below are the exact seven inputs used by the team-trained Random Forest model.</div>", unsafe_allow_html=True)
    feature_choice = st.selectbox("Inspect a feature",FEATURES,format_func=lambda k:FEATURE_LABELS[k])
    st.metric(FEATURE_LABELS[feature_choice],f"{features[feature_choice]:,.5f}")
    st.info(FEATURE_PHYSICS[feature_choice])
    feature_df = pd.DataFrame({"Feature":[FEATURE_LABELS[k] for k in FEATURES],"Value":[features[k] for k in FEATURES]})
    fig = go.Figure(go.Bar(x=feature_df["Feature"],y=feature_df["Value"],
        marker_color=["#59e7f2" if k==feature_choice else "#49627f" for k in FEATURES]))
    fig.update_layout(xaxis_tickangle=-25)
    st.plotly_chart(chart_style(fig,"CURRENT RECORDING / FEATURE PROFILE","Extracted acoustic feature","Value",380),use_container_width=True)
    st.markdown("#### Model feature importance")
    importances = getattr(model,"feature_importances_",None)
    if importances is not None and len(importances)==len(FEATURES):
        imp = pd.DataFrame({"Feature":[FEATURE_LABELS[k] for k in FEATURES],"Importance":importances}).sort_values("Importance",ascending=True)
        fig2 = go.Figure(go.Bar(x=imp["Importance"],y=imp["Feature"],orientation="h",marker_color="#9b9cff"))
        st.plotly_chart(chart_style(fig2,"RANDOM FOREST / FEATURE IMPORTANCE","Relative importance","Feature",330),use_container_width=True)
        st.caption("Feature importance describes how the fitted forest used features in its training splits. It does not prove causation.")
    else:
        st.caption("This model object does not expose feature_importances_.")

with tab5:
    st.markdown("### Inside the AI pipeline")
    st.markdown('<div class="note"><b>Input:</b> machine WAV recording → <b>Feature extraction:</b> seven acoustic descriptors → <b>Classifier:</b> trained Random Forest → <b>Output:</b> Normal (0) or Anomalous (1).</div>',unsafe_allow_html=True)
    st.markdown("#### Dataset overview")
    table = load_feature_table()
    if table is not None:
        counts = table["Label"].value_counts().reindex([0,1],fill_value=0)
        n0,n1,n2 = st.columns(3)
        n0.metric("Feature records",f"{len(table):,}")
        n1.metric("Normal label (0)",f"{int(counts.loc[0]):,}")
        n2.metric("Anomalous label (1)",f"{int(counts.loc[1]):,}")
        pie = go.Figure(go.Pie(labels=["Normal (0)","Anomalous (1)"],values=[int(counts.loc[0]),int(counts.loc[1])],hole=.62,
            marker=dict(colors=["#59e7f2","#ff819d"]),textinfo="label+percent"))
        pie.update_layout(paper_bgcolor="rgba(0,0,0,0)",font=dict(color="#dcecff"),height=310,margin=dict(t=20,b=10,l=10,r=10),showlegend=False)
        st.plotly_chart(pie,use_container_width=True)
        st.caption("Counts are read from the supplied feature dataset. The original audio corpus is not bundled in full.")
    else:
        st.warning("Feature dataset workbook not found or its expected columns are missing.")
    st.markdown("#### What this prototype can and cannot claim")
    st.write("- It classifies recordings into the two labels represented in the supplied training data.")
    st.write("- It does not identify a specific fault type because the current model output is binary.")
    st.write("- It is not a confirmed mechanical diagnosis and should not be presented as field validation.")
    st.write("- The dashboard uses the supplied trained model; it does not retrain the model in the interface.")

st.markdown("---")
st.markdown('<div class="small" style="text-align:center">AURA NEXUS • AI ACOUSTIC FAULT DETECTION FOR MACHINES • TEAM RESEARCH PROTOTYPE</div>',unsafe_allow_html=True)
