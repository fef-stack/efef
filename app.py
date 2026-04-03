import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="고도화된 MOSFET 물리 시뮬레이터", layout="wide")

st.title("🔬 Advanced MOSFET Simulator: 산란 메커니즘 및 열화 분리")
st.markdown("---")

# ==========================================
# 0. 제어 패널 (사이드바)
# ==========================================
st.sidebar.header("🎛️ 환경 및 구조 파라미터")
T_K = st.sidebar.slider("🌡️ 동작 온도 (T) [K]", min_value=300, max_value=400, value=300, step=10)
N_A_str = st.sidebar.select_slider("🧬 채널 도핑 농도 (N_A) [cm⁻³]", options=["1e16", "5e16", "1e17", "5e17", "1e18"], value="1e17")
N_A = float(N_A_str)

st.sidebar.divider()
st.sidebar.header("⚡ 물리적 스트레스 인가")
stress_time = st.sidebar.slider("⏳ 스트레스 시간 (Years)", min_value=0.0, max_value=10.0, value=0.0, step=0.5)
stress_vd = st.sidebar.slider("⚡ 가혹 전압 (V_d,stress)", min_value=1.0, max_value=3.3, value=1.5, step=0.1)

# HCI vs NBTI 뉘앙스를 위한 트랩 비율 조절
trap_type = st.sidebar.radio("결함 생성 지배 메커니즘", ["계면 트랩 지배 (N_it)", "산화막 트랩 지배 (N_ot)"])

# ==========================================
# 1. 물리 엔진 (First-Principles based Models)
# ==========================================
# 물리 상수
q = 1.6e-19
k_B = 1.38e-23 # J/K
k_eV = 8.617e-5 # eV/K
eps_0 = 8.85e-14 # F/cm
eps_si = 11.7 * eps_0
eps_ox = 3.9 * eps_0
n_i = 1.5e10 # cm^-3 at 300K (온도 의존성은 간략화)
t_ox = 2e-7 # 2nm

# 기본 커패시턴스 및 포텐셜 계산
C_ox = eps_ox / t_ox # F/cm^2
phi_F = (k_B * T_K / q) * np.log(N_A / n_i) # 페르미 포텐셜
W_dep = np.sqrt((2 * eps_si * (2 * phi_F)) / (q * N_A)) # 최대 공핍층 두께
C_d = eps_si / W_dep # 공핍층 커패시턴스

# 초기 상태 파라미터 (Flat-band voltage 등 간략화하여 Vth0 계산)
V_th0 = 0.4 + (2 * phi_F) + (np.sqrt(2 * q * eps_si * N_A * (2 * phi_F)) / C_ox)
N_it_initial = 1e10 # 초기 계면 결함

# ------------------------------------------
# 스트레스에 의한 결함 생성 모델링
# ------------------------------------------
delta_trap_max = 0
if stress_time > 0:
    # t^0.5 의존성 (Reaction-Diffusion Model)
    delta_trap_max = 5e10 * np.exp(1.2 * stress_vd) * (stress_time ** 0.5)

# 트랩 종류 분리 (N_it vs N_ot)
if trap_type == "계면 트랩 지배 (N_it)":
    delta_Nit = delta_trap_max * 0.9
    delta_Not = delta_trap_max * 0.1
else:
    delta_Nit = delta_trap_max * 0.1
    delta_Not = delta_trap_max * 0.9

N_it_total = N_it_initial + delta_Nit
N_ot_total = delta_Not

# ------------------------------------------
# 거시적 소자 특성 열화 계산
# ------------------------------------------
# 1. Vth Shift (Nit와 Not 모두 영향)
delta_Vth = (q * (delta_Nit + delta_Not)) / C_ox
V_th_degraded = V_th0 + delta_Vth

# 2. SS (Subthreshold Swing) 계산 (온도 T, C_d, N_it만 영향)
# N_ot는 SS를 눕히지 않고 커브만 이동시킴
SS_ideal = np.log(10) * (k_B * T_K / q) * (1 + (C_d + q * N_it_initial) / C_ox)
SS_degraded = np.log(10) * (k_B * T_K / q) * (1 + (C_d + q * N_it_total) / C_ox)

# ------------------------------------------
# Id-Vg 커브 계산 (전압 의존적 이동도 적용)
# ------------------------------------------
Vg_sweep = np.linspace(0.0, 1.5, 200)
Id_ideal = np.zeros_like(Vg_sweep)
Id_degraded = np.zeros_like(Vg_sweep)
mu_eff_array = np.zeros_like(Vg_sweep)

for i, Vg in enumerate(Vg_sweep):
    # A. 이동도 모델링 (Matthiessen's Rule)
    # 1. Phonon Scattering (온도 의존성)
    mu_ph = 300 * ((300 / T_K) ** 1.5)
    
    # 2. Surface Roughness Scattering (수직 전계 의존성)
    # Vg가 커질수록 전자가 계면으로 끌려가 산란 심화
    E_eff = max(0, (Vg - V_th_degraded)) 
    mu_sr = 1000 / (1 + 2.0 * E_eff**2) if E_eff > 0 else 1000
    
    # 3. Coulomb Scattering (결함 의존성)
    # 결함이 많을수록 감소 (초기 상태 방지를 위해 분모에 작은 값 추가)
    total_defects = max(1e10, N_it_total + N_ot_total)
    mu_coulomb = 1e16 / total_defects
    
    # 유효 이동도 계산 (조화 평균)
    mu_eff = 1 / (1/mu_ph + 1/mu_sr + 1/mu_coulomb)
    mu_eff_array[i] = mu_eff
    
    # B. 전류 계산
    # Ideal Curve
    if Vg < V_th0:
        Id_ideal[i] = 1e-12 * 10 ** ((Vg - V_th0) / SS_ideal)
    else:
        Id_ideal[i] = 0.5 * mu_eff * C_ox * 5 * ((Vg - V_th0) ** 2) + 1e-12
        
    # Degraded Curve
    if Vg < V_th_degraded:
        Id_degraded[i] = 1e-12 * 10 ** ((Vg - V_th_degraded) / SS_degraded)
    else:
        Id_degraded[i] = 0.5 * mu_eff * C_ox * 5 * ((Vg - V_th_degraded) ** 2) + 1e-12

# ==========================================
# 2. 통합 시각화
# ==========================================
fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.1,
                    subplot_titles=("전압에 따른 유효 이동도 변화 (μ_eff)", 
                                    "전달 특성 (Id-Vg) 곡선 변화"))

# [왼쪽] 이동도 변화 곡선
fig.add_trace(go.Scatter(x=Vg_sweep, y=mu_eff_array, mode='lines', 
                         line=dict(color='orange', width=3),
                         name='유효 이동도 (μ_eff)'), row=1, col=1)
fig.update_xaxes(title_text="Gate Voltage (V_g) [V]", row=1, col=1)
fig.update_yaxes(title_text="Mobility [cm²/V·s]", range=[0, 400], row=1, col=1)

# [오른쪽] Id-Vg 커브
fig.add_trace(go.Scatter(x=Vg_sweep, y=Id_ideal, mode='lines', 
                         line=dict(color='gray', width=2, dash='dash'), 
                         name='초기 상태 (Fresh)'), row=1, col=2)
fig.add_trace(go.Scatter(x=Vg_sweep, y=Id_degraded, mode='lines', 
                         line=dict(color='cyan', width=4), 
                         name=f'열화 상태 ({stress_time}년)'), row=1, col=2)
fig.update_xaxes(title_text="Gate Voltage (V_g) [V]", row=1, col=2)
fig.update_yaxes(title_text="Drain Current (I_d) [A] - Log", type="log", range=[-12, -3], row=1, col=2)

fig.update_layout(height=500, template="plotly_dark", margin=dict(l=20, r=20, t=60, b=20))
st.plotly_chart(fig, use_container_width=True)

# ==========================================
# 3. 소자 상태 파라미터 추출 매트릭스
# ==========================================
st.divider()
st.subheader("📊 엄밀한 물리 모델 기반 추출 파라미터")
m1, m2, m3, m4 = st.columns(4)

m1.metric("초기 문턱 전압 (V_th0)", f"{V_th0:.3f} V", 
          f"N_A 기반 계산", delta_color="off")
m2.metric("문턱 전압 변동 (ΔV_th)", f"+{delta_Vth*1000:.1f} mV", 
          f"N_it & N_ot 영향", delta_color="inverse")
m3.metric("Subthreshold Swing (SS)", f"{SS_degraded * 1000:.1f} mV/dec", 
          f"+{(SS_degraded - SS_ideal)*1000:.1f} mV (N_it만 영향)", delta_color="inverse")
I_on_drop_pct = ((Id_ideal[-1] - Id_degraded[-1]) / Id_ideal[-1]) * 100 if Id_ideal[-1] > 0 else 0
m4.metric("최대 I_on 감소율", f"-{I_on_drop_pct:.1f} %", 
          f"Mobility & V_th 열화", delta_color="inverse")
