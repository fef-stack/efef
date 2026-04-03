import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="고도화된 통합 물리 시뮬레이터", layout="wide")

st.title("🔬 통합 소자 열화 시뮬레이터 (Cross-Coupled Physics)")
st.markdown("---")

# ==========================================
# 1. 제어 패널 (이 부분이 반드시 계산보다 먼저 와야 합니다!)
# ==========================================
st.sidebar.header("🎛️ 환경 및 구조 파라미터")
T_K = st.sidebar.slider("🌡️ 동작 온도 (T) [K]", 300, 400, 300, 10)
N_A_str = st.sidebar.select_slider("🧬 채널 도핑 농도 (N_A) [cm⁻³]", options=["1e16", "5e16", "1e17", "5e17", "1e18"], value="1e17")
N_A = float(N_A_str)

st.sidebar.divider()
st.sidebar.header("⚡ 스트레스 및 트랩 밀도")
stress_time = st.sidebar.slider("⏳ 스트레스 시간 (Years)", 0.0, 10.0, 0.0, 0.5)
stress_vd = st.sidebar.slider("⚡ 가혹 전압 (V_d,stress) [V]", 1.0, 3.3, 1.5, 0.1)

st.sidebar.markdown("### 🪤 개별 트랩 수동 제어")
st.sidebar.caption("스트레스 조건에 의해 기본값이 자동 계산되며, 수동으로 튜닝하여 영향을 비교할 수 있습니다.")

# 스트레스 모델에 의한 기본 트랩 계산 (Coupling 물리 모델 적용)
q = 1.6e-19
k_B = 1.38e-23
k_eV = 8.617e-5 # eV/K (Arrhenius equation 용)

if stress_time > 0:
    # 1. HCI 메커니즘 (저온일수록, N_A가 높을수록 수평전계가 강해져 악화)
    electric_field_factor = (N_A / 1e17) ** 0.5 
    hci_temp_factor = (300 / T_K) ** 1.5 
    hci_trap = 2e10 * np.exp(1.0 * stress_vd) * (stress_time ** 0.5) * electric_field_factor * hci_temp_factor

    # 2. NBTI 메커니즘 (고온일수록 악화 - Arrhenius Model)
    E_a = 0.15 
    bti_temp_factor = np.exp(-(E_a / k_eV) * (1/T_K - 1/300))
    bti_trap = 3e10 * np.exp(1.2 * stress_vd) * (stress_time ** 0.5) * bti_temp_factor

    delta_trap_base = hci_trap + bti_trap
else:
    delta_trap_base = 0

Nit_slider = st.sidebar.slider("계면 트랩 밀도 (N_it) [x 10¹¹ cm⁻²]", 0.1, 50.0, max(0.1, delta_trap_base*0.8/1e11), 0.1)
Not_slider = st.sidebar.slider("산화막 트랩 밀도 (N_ot) [x 10¹¹ cm⁻²]", 0.0, 50.0, delta_trap_base*0.2/1e11, 0.1)

# 실제 적용될 트랩 밀도
N_it_total = 1e10 + (Nit_slider * 1e11) # 기본 1e10 내재
N_ot_total = (Not_slider * 1e11)

# ==========================================
# 2. 물리 엔진 연산 (Cross-Coupled Physics 반영)
# ==========================================
eps_0 = 8.85e-14
eps_si = 11.7 * eps_0
eps_ox = 3.9 * eps_0
n_i = 1.5e10 
t_ox = 2e-7 
C_ox = eps_ox / t_ox

# [Coupling 1] N_A -> V_th 및 Q_dep 계산
phi_F = (k_B * T_K / q) * np.log(N_A / n_i)
Q_dep = np.sqrt(2 * q * eps_si * N_A * (2 * phi_F)) # 공핍층 전하량
W_dep = Q_dep / (q * N_A)
C_d = eps_si / W_dep

V_th0 = 0.4 + (2 * phi_F) + (Q_dep / C_ox)

# 트랩 열화 반영
delta_Vth = (q * ((N_it_total - 1e10) + N_ot_total)) / C_ox
V_th_degraded = V_th0 + delta_Vth

SS_ideal = np.log(10) * (k_B * T_K / q) * (1 + (C_d + q * 1e10) / C_ox)
SS_degraded = np.log(10) * (k_B * T_K / q) * (1 + (C_d + q * N_it_total) / C_ox)

# I-V 및 Mobility 커브 연산
Vg_sweep = np.linspace(0.0, 1.5, 200)
Id_ideal, Id_degraded, mu_eff_array = [], [], []

for Vg in Vg_sweep:
    # [Coupling 3] N_A & Vg -> 수직 전계(E_eff) -> 표면 거칠기 산란(mu_sr)
    Q_inv_ideal = C_ox * max(0, Vg - V_th0)
    Q_inv_degraded = C_ox * max(0, Vg - V_th_degraded)
    
    E_eff_ideal = (Q_dep + 0.5 * Q_inv_ideal) / eps_si
    E_eff_degraded = (Q_dep + 0.5 * Q_inv_degraded) / eps_si

    # Matthiessen's Rule 이동도
    mu_ph = 300 * ((300 / T_K) ** 1.5)
    mu_sr = 1000 / (1 + (E_eff_degraded * 1e-5)**2) 
    mu_coulomb = 1e16 / max(1e10, N_it_total + N_ot_total)
    
    mu_eff = 1 / (1/mu_ph + 1/mu_sr + 1/mu_coulomb)
    mu_eff_array.append(mu_eff)
    
    # 이상적 커브
    if Vg < V_th0:
        Id_ideal.append(1e-12 * 10 ** ((Vg - V_th0) / SS_ideal))
    else:
        mu_eff_ideal = 1 / (1/mu_ph + 1/(1000 / (1 + (E_eff_ideal * 1e-5)**2)) + 1/(1e16 / 1e10))
        Id_ideal.append(0.5 * mu_eff_ideal * C_ox * 5 * ((Vg - V_th0) ** 2) + 1e-12)
        
    # 열화 커브
    if Vg < V_th_degraded:
        Id_degraded.append(1e-12 * 10 ** ((Vg - V_th_degraded) / SS_degraded))
    else:
        Id_degraded.append(0.5 * mu_eff * C_ox * 5 * ((Vg - V_th_degraded) ** 2) + 1e-12)

# ==========================================
# 3. 통합 시각화 패널 구성
# ==========================================
fig = make_subplots(
    rows=1, cols=2, 
    horizontal_spacing=0.15,
    specs=[[{"type": "xy"}, {"secondary_y": True}]],
    subplot_titles=("소자 내부 현상 (Nit & Not 직관적 시각화)", "통합 전달 특성 및 유효 이동도 (μ_eff)")
)

# --- [왼쪽] 소자 2D 단면 및 트랩 시각화 ---
fig.add_shape(type="rect", x0=0, y0=2, x1=2, y1=5, line=dict(color="cyan"), fillcolor="rgba(0,255,255,0.05)", row=1, col=1)
fig.add_annotation(x=1, y=3.5, text="Source", showarrow=False, font=dict(color="cyan"), row=1, col=1)
fig.add_shape(type="rect", x0=8, y0=2, x1=10, y1=5, line=dict(color="cyan"), fillcolor="rgba(0,255,255,0.05)", row=1, col=1)
fig.add_annotation(x=9, y=3.5, text="Drain", showarrow=False, font=dict(color="cyan"), row=1, col=1)
fig.add_shape(type="rect", x0=2.5, y0=0.5, x1=7.5, y1=1.5, line=dict(color="yellow"), fillcolor="rgba(255,255,0,0.1)", row=1, col=1)
fig.add_annotation(x=5, y=1, text="Gate", showarrow=False, font=dict(color="yellow"), row=1, col=1)
fig.add_shape(type="line", x0=2, y0=2, x1=8, y1=2, line=dict(color="white", width=2, dash="dash"), row=1, col=1)
fig.add_annotation(x=5, y=2.2, text="SiO₂ Interface", showarrow=False, font=dict(color="white", size=10), row=1, col=1)

np.random.seed(42) 
num_nit = int(Nit_slider * 2) 
if num_nit > 0:
    x_nit = np.clip(np.random.normal(7.0, 1.5, num_nit), 2.5, 7.5)
    y_nit = np.random.normal(2.0, 0.05, num_nit)
    fig.add_trace(go.Scatter(x=x_nit, y=y_nit, mode='markers', marker=dict(color='cyan', size=5, opacity=0.8), name='계면 트랩 (N_it)'), row=1, col=1)

num_not = int(Not_slider * 2)
if num_not > 0:
    x_not = np.random.uniform(2.5, 7.5, num_not)
    y_not = np.random.uniform(1.6, 1.9, num_not)
    fig.add_trace(go.Scatter(x=x_not, y=y_not, mode='markers', marker=dict(color='red', size=6, opacity=0.8), name='산화막 트랩 (N_ot)'), row=1, col=1)

fig.update_xaxes(title_text="Channel Length (μm)", range=[0, 10], row=1, col=1)
fig.update_yaxes(title_text="Depth (nm)", range=[5, 0], row=1, col=1)

# --- [오른쪽] 통합 I-V 및 이동도 (이중 Y축) ---
fig.add_trace(go.Scatter(x=Vg_sweep, y=Id_ideal, mode='lines', line=dict(color='gray', width=2, dash='dash'), name='I_d (Fresh)'), row=1, col=2, secondary_y=False)
fig.add_trace(go.Scatter(x=Vg_sweep, y=Id_degraded, mode='lines', line=dict(color='cyan', width=3), name='I_d (Degraded)'), row=1, col=2, secondary_y=False)
fig.add_trace(go.Scatter(x=Vg_sweep, y=mu_eff_array, mode='lines', line=dict(color='orange', width=4), name='유효 이동도 (μ_eff)'), row=1, col=2, secondary_y=True)

fig.update_xaxes(title_text="Gate Voltage (V_g) [V]", row=1, col=2)
fig.update_yaxes(title_text="Drain Current (I_d) [A]", type="log", range=[-12, -3], row=1, col=2, secondary_y=False)
fig.update_yaxes(title_text="Effective Mobility [cm²/V·s]", range=[0, 400], showgrid=False, row=1, col=2, secondary_y=True)

fig.update_layout(height=600, template="plotly_dark", legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5))
st.plotly_chart(fig, use_container_width=True)

# ==========================================
# 4. 소자 상태 파라미터 추출
# ==========================================
st.divider()
st.subheader("📊 실시간 파라미터 추출")
c1, c2, c3, c4 = st.columns(4)

c1.metric("초기 문턱 전압 (V_th0)", f"{V_th0:.3f} V", f"N_A: {N_A_str} 적용됨", delta_color="off")
c2.metric("문턱 전압 변동 (ΔV_th)", f"+{delta_Vth*1000:.1f} mV", "우측 이동 (성능 저하)", delta_color="inverse")
c3.metric("Subthreshold Swing (SS)", f"{SS_degraded * 1000:.1f} mV/dec", f"+{(SS_degraded - SS_ideal)*1000:.1f} mV (N_it만 영향)", delta_color="inverse")
c4.metric("최고 이동도 (Max μ_eff)", f"{max(mu_eff_array):.1f} cm²/V·s", "온도 & 산란 영향 반영", delta_color="normal")
