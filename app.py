import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="고도화된 통합 물리 시뮬레이터", layout="wide")

st.title("🔬 통합 소자 물리 시뮬레이터 (Short-Channel Effect & Leakage)")
st.markdown("---")

# ==========================================
# 1. 제어 패널 (사이드바)
# ==========================================
st.sidebar.header("📐 소자 구조 및 환경 파라미터")
L_nm = st.sidebar.slider("채널 길이 (L) [nm]", min_value=10, max_value=200, value=20, step=5, 
                         help="L을 30nm 이하로 줄이면 Gate 통제력이 상실되며 Off-current(누설 전류)가 폭발적으로 증가합니다.")
T_K = st.sidebar.slider("🌡️ 동작 온도 (T) [K]", 300, 400, 300, 10)
N_A_str = st.sidebar.select_slider("🧬 채널 도핑 농도 (N_A) [cm⁻³]", options=["1e16", "5e16", "1e17", "5e17", "1e18"], value="1e17")
N_A = float(N_A_str)

st.sidebar.divider()
st.sidebar.header("⚡ 스트레스 및 트랩 밀도")
stress_time = st.sidebar.slider("⏳ 스트레스 시간 (Years)", 0.0, 10.0, 0.0, 0.5)
stress_vd = st.sidebar.slider("⚡ 가혹 전압 (V_d,stress) [V]", 1.0, 3.3, 1.5, 0.1)

# 스트레스 모델에 의한 트랩 계산
q = 1.6e-19
k_B = 1.38e-23
k_eV = 8.617e-5

if stress_time > 0:
    electric_field_factor = (N_A / 1e17) ** 0.5 
    hci_temp_factor = (300 / T_K) ** 1.5 
    hci_trap = 2e10 * np.exp(1.0 * stress_vd) * (stress_time ** 0.5) * electric_field_factor * hci_temp_factor
    E_a = 0.15 
    bti_temp_factor = np.exp(-(E_a / k_eV) * (1/T_K - 1/300))
    bti_trap = 3e10 * np.exp(1.2 * stress_vd) * (stress_time ** 0.5) * bti_temp_factor
    delta_trap_base = hci_trap + bti_trap
else:
    delta_trap_base = 0

st.sidebar.markdown("### 🪤 개별 트랩 수동 제어")
Nit_slider = st.sidebar.slider("계면 트랩 밀도 (N_it) [x 10¹¹ cm⁻²]", 0.1, 50.0, max(0.1, delta_trap_base*0.8/1e11), 0.1)
Not_slider = st.sidebar.slider("산화막 트랩 밀도 (N_ot) [x 10¹¹ cm⁻²]", 0.0, 50.0, delta_trap_base*0.2/1e11, 0.1)

N_it_total = 1e10 + (Nit_slider * 1e11)
N_ot_total = (Not_slider * 1e11)

# ==========================================
# 2. 물리 엔진 연산 (단채널 효과 수식 추가)
# ==========================================
eps_0 = 8.85e-14
eps_si = 11.7 * eps_0
eps_ox = 3.9 * eps_0
n_i = 1.5e10 
t_ox = 2e-7 
C_ox = eps_ox / t_ox

W_cm = 100 * 1e-7  
L_cm = L_nm * 1e-7 
v_sat = 1.0e7      
mu_floor = 40.0    

# 이상적(Long Channel 기준) 물리량 계산
phi_F = (k_B * T_K / q) * np.log(N_A / n_i)
Q_dep = np.sqrt(2 * q * eps_si * N_A * (2 * phi_F)) 
W_dep = Q_dep / (q * N_A)
C_d = eps_si / W_dep

V_th0_long = 0.4 + (2 * phi_F) + (Q_dep / C_ox)
SS_ideal_long = np.log(10) * (k_B * T_K / q) * (1 + (C_d + q * 1e10) / C_ox)

# 🚨 [신규 추가] Short-Channel Effect (SCE) 모델링 🚨
# 채널이 짧아질수록 특성 길이(lambda)에 의해 게이트 통제력 상실
lambda_char = 15.0 # nm (평면형 소자의 SCE 특성 길이 가정)
sce_factor = np.exp(-L_nm / lambda_char)

# 1. Vth Roll-off 및 DIBL (문턱 전압 강하)
V_th_rolloff = 0.4 * sce_factor 
DIBL_shift = 0.1 * stress_vd * sce_factor 
V_th0 = V_th0_long - V_th_rolloff - DIBL_shift

# 2. SS Degradation (Subthreshold Swing 붕괴)
SS_ideal = SS_ideal_long * (1 + 4.0 * sce_factor)

# 트랩 열화 반영
delta_Vth = (q * ((N_it_total - 1e10) + N_ot_total)) / C_ox
V_th_degraded = V_th0 + delta_Vth
SS_degraded = SS_ideal * (1 + (C_d + q * N_it_total) / C_ox) / (1 + (C_d + q * 1e10) / C_ox)

# I-V 및 Mobility 커브 연산
Vg_sweep = np.linspace(0.5, 4.0, 400)
Id_ideal, Id_degraded, mu_eff_array = [], [], []

# 문턱 전압에서의 기준 전류 (I_th)
I_th = 1e-7 * (W_cm / L_cm)

for Vg in Vg_sweep:
    Q_inv_degraded = C_ox * max(0, Vg - V_th_degraded)
    E_eff_degraded = (Q_dep + 0.5 * Q_inv_degraded) / eps_si

    # Mobility 계산 (하한선 적용)
    mu_ph = 300 * ((300 / T_K) ** 1.5)
    mu_sr = 1000 / (1 + (E_eff_degraded * 1e-5)**2) 
    mu_coulomb = 1e16 / max(1e10, N_it_total + N_ot_total)
    
    mu_eff_bulk = 1 / (1/mu_ph + 1/mu_sr + 1/mu_coulomb)
    mu_eff = max(mu_floor, mu_eff_bulk) 
    mu_eff_array.append(mu_eff)
    
    # --- [Degraded 커브 계산] ---
    if Vg < V_th_degraded:
        # Subthreshold 영역 (SCE로 인해 SS가 망가지면 누설 전류 폭발)
        I_sub = I_th * 10 ** ((Vg - V_th_degraded) / SS_degraded)
        Id_degraded.append(I_sub)
    else:
        # 속도 포화(Velocity Saturation) 반영 단채널 전류 모델
        I_long = 0.5 * mu_eff * C_ox * (W_cm / L_cm) * ((Vg - V_th_degraded) ** 2)
        theta_sat = mu_eff * (Vg - V_th_degraded) / (2 * v_sat * L_cm)
        Id_degraded.append((I_long / (1 + theta_sat)) + I_th)

    # --- [Ideal (Fresh) 커브 계산] ---
    if Vg < V_th0:
        I_sub_ideal = I_th * 10 ** ((Vg - V_th0) / SS_ideal)
        Id_ideal.append(I_sub_ideal)
    else:
        Q_inv_ideal = C_ox * max(0, Vg - V_th0)
        E_eff_ideal = (Q_dep + 0.5 * Q_inv_ideal) / eps_si
        mu_sr_ideal = 1000 / (1 + (E_eff_ideal * 1e-5)**2)
        mu_eff_bulk_ideal = 1 / (1/mu_ph + 1/mu_sr_ideal + 1/(1e16 / 1e10))
        mu_eff_ideal = max(mu_floor, mu_eff_bulk_ideal)
        
        I_long_ideal = 0.5 * mu_eff_ideal * C_ox * (W_cm / L_cm) * ((Vg - V_th0) ** 2)
        theta_sat_ideal = mu_eff_ideal * (Vg - V_th0) / (2 * v_sat * L_cm)
        Id_ideal.append((I_long_ideal / (1 + theta_sat_ideal)) + I_th)

# ==========================================
# 3. 통합 시각화 패널 구성
# ==========================================
fig = make_subplots(
    rows=1, cols=2, 
    horizontal_spacing=0.15,
    specs=[[{"type": "xy"}, {"secondary_y": True}]],
    subplot_titles=(f"소자 내부 구조 (L = {L_nm} nm)", "통합 전달 특성 (단채널 누설 붕괴 시각화)")
)

# [왼쪽] 소자 단면 시각화 (스케일링 반영)
S_end = L_nm * 0.2
D_start = L_nm * 0.8
G_start = L_nm * 0.25
G_end = L_nm * 0.75

fig.add_shape(type="rect", x0=0, y0=2, x1=S_end, y1=5, line=dict(color="cyan"), fillcolor="rgba(0,255,255,0.05)", row=1, col=1)
fig.add_annotation(x=S_end/2, y=3.5, text="Source", showarrow=False, font=dict(color="cyan"), row=1, col=1)
fig.add_shape(type="rect", x0=D_start, y0=2, x1=L_nm, y1=5, line=dict(color="cyan"), fillcolor="rgba(0,255,255,0.05)", row=1, col=1)
fig.add_annotation(x=D_start + (L_nm-D_start)/2, y=3.5, text="Drain", showarrow=False, font=dict(color="cyan"), row=1, col=1)
fig.add_shape(type="rect", x0=G_start, y0=0.5, x1=G_end, y1=1.5, line=dict(color="yellow"), fillcolor="rgba(255,255,0,0.1)", row=1, col=1)
fig.add_annotation(x=L_nm/2, y=1, text="Gate", showarrow=False, font=dict(color="yellow"), row=1, col=1)
fig.add_shape(type="line", x0=S_end, y0=2, x1=D_start, y1=2, line=dict(color="white", width=2, dash="dash"), row=1, col=1)

np.random.seed(42) 
num_nit = int(Nit_slider * 2) 
if num_nit > 0:
    x_nit = np.clip(np.random.normal(L_nm*0.7, L_nm*0.1, num_nit), G_start, G_end)
    y_nit = np.random.normal(2.0, 0.05, num_nit)
    fig.add_trace(go.Scatter(x=x_nit, y=y_nit, mode='markers', marker=dict(color='cyan', size=5, opacity=0.8), name='N_it'), row=1, col=1)

num_not = int(Not_slider * 2)
if num_not > 0:
    x_not = np.random.uniform(G_start, G_end, num_not)
    y_not = np.random.uniform(1.6, 1.9, num_not)
    fig.add_trace(go.Scatter(x=x_not, y=y_not, mode='markers', marker=dict(color='red', size=6, opacity=0.8), name='N_ot'), row=1, col=1)

fig.update_xaxes(title_text="Channel Position (nm)", range=[0, L_nm], row=1, col=1)
fig.update_yaxes(title_text="Depth (nm)", range=[5, 0], row=1, col=1)

# [오른쪽] 통합 I-V 및 이동도 (Log 범위 대폭 확장)
fig.add_trace(go.Scatter(x=Vg_sweep, y=Id_ideal, mode='lines', line=dict(color='gray', width=2, dash='dash'), name='I_d (Fresh)'), row=1, col=2, secondary_y=False)
fig.add_trace(go.Scatter(x=Vg_sweep, y=Id_degraded, mode='lines', line=dict(color='cyan', width=3), name='I_d (Degraded)'), row=1, col=2, secondary_y=False)
fig.add_trace(go.Scatter(x=Vg_sweep, y=mu_eff_array, mode='lines', line=dict(color='orange', width=4), name='유효 이동도 (μ_eff)'), row=1, col=2, secondary_y=True)

fig.update_xaxes(title_text="Gate Voltage (V_g) [V]", row=1, col=2)
# Leakage를 관찰하기 위해 Y축 하한을 10^-12 고정, 상한은 10^-2
fig.update_yaxes(title_text="Drain Current (I_d) [A]", type="log", range=[-12, -2], row=1, col=2, secondary_y=False)
fig.update_yaxes(title_text="Effective Mobility [cm²/V·s]", range=[0, 400], showgrid=False, row=1, col=2, secondary_y=True)

fig.update_layout(height=600, template="plotly_dark", legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5))
st.plotly_chart(fig, use_container_width=True)

# ==========================================
# 4. 소자 상태 파라미터 추출
# ==========================================
st.divider()
st.subheader("📊 실시간 소자 성능 및 누설 파라미터")
c1, c2, c3, c4 = st.columns(4)

I_off = Id_degraded[0] # Vg=0 일 때의 누설 전류
I_on = Id_degraded[-1] # Vg=1.5 일 때의 구동 전류
Ion_Ioff_ratio = I_on / I_off

c1.metric("초기 문턱 전압 (V_th0)", f"{V_th0:.3f} V", f"Roll-off & DIBL 반영", delta_color="off")
c2.metric("Subthreshold Swing (SS)", f"{SS_degraded * 1000:.1f} mV/dec", f"Gate 통제력 상실도", delta_color="inverse")
c3.metric("Off-Current (누설 전류)", f"{I_off:.1e} A", "Vg = 0V 기준 누설량", delta_color="inverse")

# 점멸비가 10^4 이하면 붉은색 경고 느낌을 주기 위한 처리
ratio_str = f"10^{np.log10(Ion_Ioff_ratio):.1f}"
c4.metric("I_on / I_off 점멸비", ratio_str, "10^4 이하 시 스위치 기능 상실", delta_color="normal" if Ion_Ioff_ratio > 1e4 else "inverse")
