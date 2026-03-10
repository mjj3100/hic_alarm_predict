import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# 기본 설정
# -----------------------------
st.set_page_config(
    page_title="HIC Alarm Prediction Tool",
    page_icon="🧪",
    layout="wide"
)

# -----------------------------
# 스타일
# -----------------------------
st.markdown("""
    <style>
    .main {
        background-color: #f7f8fa;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    h1, h2, h3 {
        color: #1f2937;
    }
    .sub-box {
        background-color: #ffffff;
        padding: 1.2rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        margin-bottom: 1rem;
    }
    .result-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #374151;
    }
    .small-text {
        color: #6b7280;
        font-size: 0.9rem;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------
# 모델 불러오기
# -----------------------------
model = joblib.load("alarm_prediction_model.pkl")

# -----------------------------
# 함수 정의
# -----------------------------
def predict_alarm_free_prob(protein, temperature):
    X = np.array([[protein, temperature]])
    prob_alarm_free = model.predict_proba(X)[0][1]
    prob_alarm = model.predict_proba(X)[0][0]
    return prob_alarm_free, prob_alarm


def get_zone(prob_alarm_free):
    if prob_alarm_free >= 0.8:
        return "SAFE", "success"
    elif prob_alarm_free >= 0.5:
        return "CAUTION", "warning"
    else:
        return "RISK", "error"


def recommend_temperature_range(protein, threshold=0.8, t_min=15, t_max=25, n=300):
    temps = np.linspace(t_min, t_max, n)
    probs = []

    for t in temps:
        p, _ = predict_alarm_free_prob(protein, t)
        probs.append(p)

    probs = np.array(probs)
    safe_temps = temps[probs >= threshold]

    if len(safe_temps) == 0:
        return None, temps, probs

    return (safe_temps.min(), safe_temps.max()), temps, probs


def draw_safe_zone_map(protein_input, temp_input):
    protein_range = np.linspace(8200, 9500, 150)
    temp_range = np.linspace(15, 25, 150)

    P, T = np.meshgrid(protein_range, temp_range)
    grid = np.c_[P.ravel(), T.ravel()]
    Z = model.predict_proba(grid)[:, 1].reshape(P.shape)

    fig, ax = plt.subplots(figsize=(7, 5))

    contour = ax.contourf(P, T, Z, levels=np.linspace(0, 1, 21), cmap="viridis")
    cbar = fig.colorbar(contour, ax=ax)
    cbar.set_label("P(Alarm = 0)")

    # Safe boundary
    boundary = ax.contour(P, T, Z, levels=[0.8], colors="red", linewidths=2)
    ax.clabel(boundary, fmt="P=0.8 Safe Boundary", inline=True, fontsize=9)

    # 현재 입력값
    current_prob, _ = predict_alarm_free_prob(protein_input, temp_input)
    ax.scatter(
        protein_input,
        temp_input,
        marker="*",
        s=300,
        c="cyan",
        edgecolors="black",
        linewidths=1.5,
        label=f"Input (P0={current_prob:.0%})"
    )

    ax.set_title("Safe Zone Map")
    ax.set_xlabel("TK36 Protein")
    ax.set_ylabel("Loading Temperature (°C)")
    ax.legend(loc="upper right")

    return fig


def draw_temperature_curve(protein, current_temp):
    temps = np.linspace(15, 25, 300)
    probs = []

    for t in temps:
        p, _ = predict_alarm_free_prob(protein, t)
        probs.append(p)

    probs = np.array(probs)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(temps, probs, linewidth=2, label=f"Protein = {protein}")
    ax.axhline(0.8, linestyle="--", color="red", label="Safe Threshold (0.8)")
    ax.axvline(current_temp, linestyle=":", color="black", label=f"Current Temp = {current_temp:.1f}")

    ax.set_title("Temperature vs Alarm-Free Probability")
    ax.set_xlabel("Loading Temperature (°C)")
    ax.set_ylabel("P(Alarm = 0)")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    ax.legend()

    return fig


# -----------------------------
# 헤더
# -----------------------------
st.title("🧪 HIC Alarm Prediction Tool")
st.write("TK36 Protein과 Loading Temperature를 입력하면 **Alarm = 0 확률**, **Zone 판정**, **권장 온도 범위**를 예측합니다.")

# -----------------------------
# 입력 영역
# -----------------------------
with st.container():
    st.markdown('<div class="sub-box">', unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        protein = st.number_input(
            "TK36 Protein",
            min_value=8000,
            max_value=10000,
            value=9000,
            step=10
        )

    with col2:
        temperature = st.number_input(
            "Loading Temperature (°C)",
            min_value=15.0,
            max_value=25.0,
            value=20.0,
            step=0.1
        )

    with col3:
        threshold = st.slider(
            "Safe Threshold",
            min_value=0.50,
            max_value=0.95,
            value=0.80,
            step=0.05
        )

    run_button = st.button("🚀 Run Prediction", use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# 결과 영역
# -----------------------------
if run_button:
    prob_alarm_free, prob_alarm = predict_alarm_free_prob(protein, temperature)
    zone_text, zone_type = get_zone(prob_alarm_free)

    temp_range, temps, probs = recommend_temperature_range(
        protein=protein,
        threshold=threshold,
        t_min=15,
        t_max=25
    )

    # KPI 표시
    st.subheader("Prediction Result")

    k1, k2, k3 = st.columns(3)

    with k1:
        st.metric("Alarm = 0 Probability", f"{prob_alarm_free*100:.1f}%")

    with k2:
        st.metric("Alarm Probability", f"{prob_alarm*100:.1f}%")

    with k3:
        st.metric("Zone", zone_text)

    # Zone 메시지
    if zone_type == "success":
        st.success("현재 입력 조건은 SAFE ZONE입니다.")
    elif zone_type == "warning":
        st.warning("현재 입력 조건은 CAUTION ZONE입니다. 온도 조건 조정이 필요할 수 있습니다.")
    else:
        st.error("현재 입력 조건은 RISK ZONE입니다. 공정 조건 재검토가 필요합니다.")

    # 추천 온도 범위
    st.subheader("Recommended Temperature Range")

    if temp_range is None:
        st.error(f"P(Alarm = 0) ≥ {threshold:.2f} 를 만족하는 온도 범위를 찾지 못했습니다.")
    else:
        low, high = temp_range
        st.info(f"권장 Loading Temperature 범위: **{low:.2f} ~ {high:.2f} °C**")

    # 그래프 영역
    left, right = st.columns(2)

    with left:
        st.subheader("Safe Zone Map")
        fig1 = draw_safe_zone_map(protein, temperature)
        st.pyplot(fig1)

    with right:
        st.subheader("Temperature Requirement Curve")
        fig2 = draw_temperature_curve(protein, temperature)
        st.pyplot(fig2)

    # 부가 설명
    st.markdown("---")
    st.markdown("### Interpretation")
    st.write(
        f"- 현재 입력값 기준 Alarm = 0 확률은 **{prob_alarm_free*100:.1f}%** 입니다.\n"
        f"- Zone 판정 결과는 **{zone_text}** 입니다.\n"
        f"- 입력한 Protein 조건에서 안정적인 운영을 위해 추천되는 온도 범위를 함께 제시합니다."
    )

else:
    st.info("좌측 입력값을 설정한 뒤 **Run Prediction** 버튼을 눌러주세요.")