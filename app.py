import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="고도화된 통합 물리 시뮬레이터", layout="wide")

st.title("🔬 통합 소자 물리 시뮬레이터 (Short-Channel Effect & Reliability)")

# 🚨 [m11 반영] Disclaimer 배너 (UI 최상단 고정)
st.warning("⚠️ **Disclaimer:** This simulator is a qualitative educational surrogate model, not a calibrated TCAD solver. Quantitative values (e.g., absolute ΔV_th in mV) should not be taken as device-accurate predictions.")

st.markdown("본 시뮬레이터는 전공 입문자가 반도체 소자의 단채널 효과(SCE)와 열화 경향(HCI, BTI)을 직관적으로 탐색할 수 있도록 설계된 **정성적 교육용 대리 모델(Qualitative Educational Surrogate Model)**입니다.")
st.markdown("---")

# ==========================================
# 1. 제어 패널 (사이드바)
# ==========================================
st.sidebar.header("📐 측정 환경 (Measurement Phase)")
L_nm = st.sidebar.slider("채널 길이 (L) [nm]", min_value=10, max_value=200, value=20, step=5)
T_K = st.sidebar.slider("🌡️ 동작 온도 (T) [K]", 300, 400, 300, 10)
N_A = float(st.sidebar.select_slider("🧬 채널 도핑 농도 (N_A) [cm⁻³]", options=["1e16", "5e16", "1e17", "5e17", "1e18"], value="1e17"))
V_d_read = st.sidebar.slider("측정 드레인 전압 (V_d,read) [V]", 0.1, 2.0, 1.0, 0.1)

st.sidebar.divider()
st.sidebar.header("⚡ 물리적 효과 제어 (Attribution)")
apply_sce = st.sidebar.checkbox("단채널 효과 (SCE) 적용", value=True)
apply_hci = st.sidebar.checkbox("HCI 열화 모델 적용", value=True)
apply_bti = st.sidebar.checkbox("BTI 열화 모델 적용", value=True)

st.sidebar.divider()
st.sidebar.header("🔥 스트레스 인가 (Stress Phase)")
stress_time = st.sidebar.slider("⏳ 스트레스 시간 (Years)", 0.0, 10.0, 0.0, 0.5)
stress_vd = st.sidebar.slider("⚡ 가혹 전압 (V_d,stress) [V]", 1.0, 3.3, 2.0, 0.1)

# 🎲 [m10 반영] 확률적 제어 (Stochastic Control)
st.sidebar.divider()
st.sidebar.header("🎲 확률적 제어 (Stochastic Control)")
random_seed = st.sidebar.number_input("Random Seed (시드 고정)", min_value=0, max_value=9999, value=42, step=1)
ensemble_mode = st.sidebar.toggle("Ensemble (N=20) Average", value=False, help="20개의 무작위 트랩 노이즈 결과를 앙상블 평균합니다.")

# ==========================================
# 2. 물리 엔진 연산 (함수화 적용)
# ==========================================
q, k_B, k_eV = 1.6e-19, 1.38e-23, 8.617e-5
eps_0 = 8.85e-14
eps_si = 11.7 * eps_0
eps_ox = 3.9 * eps_0
n_i = 1.5e10 
t_ox = 2e-7 
C_ox = eps_ox / t_ox
W_cm, L_cm, v_sat, mu_floor = 1e-5, L_nm * 1e-7, 1.0e7, 40.0    
baseline_Nit = 1e10

# 트랩 기본 연산
hci_trap, bti_trap = 0, 0
if stress_time > 0:
    if apply_hci:
        hci_trap = 2e10 * np.exp(1.0 * stress_vd) * (stress_time ** 0.5) * ((N_A/1e17)**0.5) * ((300/T_K)**1.5)
    if apply_bti:
        bti_trap = 3e10 * np.exp(1.2 * stress_vd) * (stress_time ** 0.5) * np.exp(-(0.15/k_eV) * (1/T_K - 1/300))

mean_delta_Nit = (hci_trap + bti_trap) * 0.8
mean_delta_Not = (hci_trap + bti_trap) * 0.2

# Fresh 상태 공통 연산
phi_F = (k_B * T_K / q) * np.log(N_A / n_i)
Q_dep = np.sqrt(2 * q * eps_si * N_A * (2 * phi_F)) 
W_dep = Q_dep / (q * N_A)
C_d = eps_si / W_dep

V_th0_long = 0.4 + (2 * phi_F) + (Q_dep / C_ox)
SS_ideal_long = np.log(10) * (k_B * T_K / q) * (1 + (C_d + q * baseline_Nit) / C_ox)

lambda_char = 15.0 
sce_factor = np.exp(-L_nm / lambda_char) if apply_sce else 0.0
V_th_fresh = V_th0_long - (0.4 * sce_factor) - (0.1 * V_d_read * sce_factor)
SS_fresh = SS_ideal_long * (1 + 4.0 * sce_factor)

Vg_sweep = np.linspace(0.0, 4.0, 200)
I_th = 1e-7 * (W_cm / L_cm)

def compute_Id(Nit_val, Not_val):
    Id_array = []
    delta_Vth = (q * (Nit_val + Not_val)) / C_ox
    V_th_deg = V_th_fresh + delta_Vth
    SS_deg = SS_fresh * (1 + (C_d + q * (baseline_Nit + Nit_val)) / C_ox) / (1 + (C_d + q * baseline_Nit) / C_ox)
    
    for Vg in Vg_sweep:
        Q_inv = C_ox * max(0, Vg - V_th_deg)
        E_eff = (Q_dep + 0.5 * Q_inv) / eps_si
        mu_ph = 300 * ((300 / T_K) ** 1.5)
        mu_sr = 1000 / (1 + (E_eff * 1e-5)**2) 
        mu_coulomb = 1e16 / max(1e10, baseline_Nit + Nit_val + Not_val)
        mu_eff = max(mu_floor, 1 / (1/mu_ph + 1/mu_sr + 1/mu_coulomb))
        
        if Vg < V_th_deg:
            Id_array.append(I_th * 10 ** ((Vg - V_th_deg) / max(1e-6, SS_deg)))
        else:
            I_long = 0.5 * mu_eff * C_ox * (W_cm / L_cm) * ((Vg - V_th_deg) ** 2)
            theta_sat = mu_eff * (Vg - V_th_deg) / (2 * v_sat * L_cm)
            Id_array.append((I_long / (1 + theta_sat)) + I_th)
    return np.array(Id_array), V_th_deg, SS_deg

# 🎲 [m10 반영] 앙상블 평균 또는 단일 난수 연산
Id_fresh, _, _ = compute_Id(0, 0)
np.random.seed(random_seed)

if ensemble_mode:
    Id_ensemble = []
    for _ in range(20):
        noise = np.random.normal(1.0, 0.1) # 10% 노이즈
        n_it_noisy, n_ot_noisy = max(0, mean_delta_Nit * noise), max(0, mean_delta_Not * noise)
        Id_temp, _, _ = compute_Id(n_it_noisy, n_ot_noisy)
        Id_ensemble.append(Id_temp)
    Id_degraded = np.mean(Id_ensemble, axis=0)
    _, V_th_degraded, SS_degraded = compute_Id(mean_delta_Nit, mean_delta_Not) # Representative params
else:
    noise = np.random.normal(1.0, 0.05) # 5% 노이즈
    Nit_total_stochastic = max(0, mean_delta_Nit * noise)
    Not_total_stochastic = max(0, mean_delta_Not * noise)
    Id_degraded, V_th_degraded, SS_degraded = compute_Id(Nit_total_stochastic, Not_total_stochastic)

# ==========================================
# 3. 통합 시각화 패널 구성 (Subplots)
# ==========================================
# 💡 [m9 반영] 보조 Y축 (secondary_y) 설정 추가
fig = make_subplots(
    rows=1, cols=2, horizontal_spacing=0.15,
    specs=[[{"secondary_y": True}, {"secondary_y": True}]],
    subplot_titles=(f"Trap Marker & E-Field (L = {L_nm} nm)", "통합 전달 특성 (단채널 & 누설)")
)

# [왼쪽] 소자 내부 구조 및 트랩 분포
S_end, D_start, G_start, G_end = L_nm*0.2, L_nm*0.8, L_nm*0.25, L_nm*0.75

fig.add_shape(type="rect", x0=0, y0=2, x1=S_end, y1=5, fillcolor="rgba(0,255,255,0.05)", line_width=0, row=1, col=1)
fig.add_shape(type="rect", x0=D_start, y0=2, x1=L_nm, y1=5, fillcolor="rgba(0,255,255,0.05)", line_width=0, row=1, col=1)
fig.add_shape(type="rect", x0=G_start, y0=0.5, x1=G_end, y1=1.5, fillcolor="rgba(255,255,0,0.1)", line_width=0, row=1, col=1)

# 난수 재설정을 통해 시각적 트랩 배치
np.random.seed(random_seed)
num_nit = int((mean_delta_Nit/1e11) * 2) 
if num_nit > 0:
    # 드레인 쪽에 치우치도록 분포 가중치 (m9 물리적 일관성)
    x_nit = np.clip(np.random.normal(D_start - L_nm*0.1, L_nm*0.05, num_nit), G_start, G_end)
    y_nit = np.random.normal(2.0, 0.05, num_nit)
    fig.add_trace(go.Scatter(x=x_nit, y=y_nit, mode='markers', marker=dict(color='cyan', size=5), name='N_it'), row=1, col=1, secondary_y=False)

# 💡 [m9 반영] Lateral E-field 프로파일 곡선 오버레이
x_array = np.linspace(0, L_nm, 100)
# 드레인 쪽으로 갈수록 지수적으로 증가하는 E-field 단순화 모델
E_field_profile = np.exp((x_array - D_start) / lambda_char) * stress_vd
fig.add_trace(go.Scatter(x=x_array, y=E_field_profile, mode='lines', line=dict(color='magenta', width=2, dash='dot'), name='Lateral E-field (a.u.)'), row=1, col=1, secondary_y=True)

fig.update_xaxes(title_text="Channel Position (nm)", range=[0, L_nm], row=1, col=1)
fig.update_yaxes(title_text="Depth (nm)", range=[5, 0], row=1, col=1, secondary_y=False)
fig.update_yaxes(title_text="E-field Intensity", showgrid=False, row=1, col=1, secondary_y=True)

# [오른쪽] 통합 I-V (앙상블 모드 시 희미한 백그라운드 선 추가 가능, 여기선 깔끔하게 평균만 표시)
fig.add_trace(go.Scatter(x=Vg_sweep, y=Id_fresh, mode='lines', line=dict(color='gray', width=2, dash='dash'), name='I_d (Fresh)'), row=1, col=2, secondary_y=False)
fig.add_trace(go.Scatter(x=Vg_sweep, y=Id_degraded, mode='lines', line=dict(color='red', width=3), name='I_d (Degraded Average)' if ensemble_mode else 'I_d (Degraded)'), row=1, col=2, secondary_y=False)

fig.update_xaxes(title_text="Gate Voltage (V_g) [V]", row=1, col=2)
fig.update_yaxes(title_text="Drain Current (I_d) [A] [log]", type="log", range=[-12, -2], row=1, col=2, secondary_y=False)

fig.update_layout(height=500, template="plotly_dark", legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5))
st.plotly_chart(fig, use_container_width=True)
