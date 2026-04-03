import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="고도화된 통합 물리 시뮬레이터", layout="wide")

st.title("🔬 통합 소자 물리 시뮬레이터 (Short-Channel & Mobility Floor)")
st.markdown("---")

# ==========================================
# 1. 제어 패널 (사이드바)
# ==========================================
st.sidebar.header("📐 소자 구조 및 환경 파라미터")
L_nm = st.sidebar.slider("채널 길이 (L) [nm]", min_value=20, max_value=1000, value=20, step=10, 
                         help="20nm로 줄이면 속도 포화(Velocity Saturation)에 의해 전류 커브가 Linear해지고, 1000nm(Long Channel)로 늘리면 2차 함수(Quadratic)로 복귀합니다.")
T_K = st.sidebar.slider("🌡️ 동작 온도 (T) [K]", 300, 400, 300, 10)
N_A_str = st.sidebar.select_slider("🧬 채널 도핑 농도 (N_A) [cm⁻³]", options=["1e16", "5e16", "1e17", "5e17", "1e18"], value="1e17")
N_A = float(N_A_str)

st.sidebar.divider()
st.sidebar.header("⚡ 스트레스 및 트랩 밀도")
stress_time = st.sidebar.slider("⏳ 스트레스 시간 (Years)", 0.0, 10.0, 0.0, 0.5)
stress_vd = st.sidebar.slider("⚡ 가혹 전압 (V_d,stress) [V]", 1.0, 3.3, 1.5, 0.1)

# 스트레스 모델에 의한 트랩 계산 (Coupling 물리 모델)
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
# 2. 물리 엔진 연산 (속도 포화 및 Mobility Floor 반영)
# ==========================================
eps_0 = 8.85e-14
eps_si = 11.7 * eps_0
eps_ox = 3.9 * eps_0
n_i = 1.5e10 
t_ox = 2e-7 
C_ox = eps_ox / t_ox

# 차원 스케일링
W_cm = 100 * 1e-7  # 폭은 100nm 고정
L_cm = L_nm * 1e-7 # 사용자가 조절한 채널 길이 (cm)
v_sat = 1.0e7      # 전자 포화 속도 (cm/s)
mu_floor = 40.0    # 양자역학적 이동도 하한선 (cm²/V·s)

# Vth 및 Q_dep 계산
phi_F = (k_B * T_K / q) * np.log(N_A / n_i)
Q_dep = np.sqrt(2 * q * eps_si * N_A * (2 * phi_F)) 
W_dep = Q_dep / (q * N_A)
C_d = eps_si / W_dep

V_th0 = 0.4 + (2 * phi_F) + (Q_dep / C_ox)

# 트랩 열화 반영
delta_Vth = (q * ((N_it_total - 1e10) + N_ot_total)) / C_ox
V_th_degraded = V_th0 + delta_Vth

SS_ideal = np.log(10) * (k_B * T_K / q) * (1 + (C_d + q * 1e10) / C_ox)
SS_degraded = np.log(10) * (k_B * T_K / q) * (1 + (C_d + q * N_it_total) / C_ox)

# I-V 및 Mobility 커브 연산
Vg_sweep = np.linspace(1.5, 4.0, 200)
Id_ideal, Id_degraded, mu_eff_array = [], [], []

for Vg in Vg_sweep:
    Q_inv_ideal = C_ox * max(0, Vg - V_th0)
    Q_inv_degraded = C_ox * max(0, Vg - V_th_degraded)
    
    E_eff_ideal = (Q_dep + 0.5 * Q_inv_ideal) / eps_si
    E_eff_degraded = (Q_dep + 0.5 * Q_inv_degraded) / eps_si

    # [이동도 하한선(Floor) 적용]
    mu_ph = 300 * ((300 / T_K) ** 1.5)
    mu_sr = 1000 / (1 + (E_eff_degraded * 1e-5)**2) 
    mu_coulomb = 1e16 / max(1e10, N_it_total + N_ot_total)
    
    # 조화 평균 후 max() 함수로 40 밑으로 떨어지지 않게 방어
    mu_eff_bulk = 1 / (1/mu_ph + 1/mu_sr + 1/mu_coulomb)
    mu_eff = max(mu_floor, mu_eff_bulk) 
    mu_eff_array.append(mu_eff)
    
    # [속도 포화(Velocity Saturation)가 반영된 단채널 전류 모델]
    # Long Channel 전류 (2차 함수형)
    if Vg < V_th_degraded:
        Id_degraded.append(1e-12 * 10 ** ((Vg - V_th_degraded) / SS_degraded))
    else:
        I_long = 0.5 * mu_eff * C_ox * (W_cm / L_cm) * ((Vg - V_th_degraded) ** 2)
        # 단채널 감쇄 팩터: L이 작을수록 분모가 커져 전류를 선형(Linear)으로 만듦
        theta_sat = mu_eff * (Vg - V_th_degraded) / (2 * v_sat * L_cm)
        Id_degraded.append((I_long / (1 + theta_sat)) + 1e-12)

    # 이상적 상태 계산 (위와 동일 로직)
    if Vg < V_th0:
        Id_ideal.append(1e-12 * 10 ** ((Vg - V_th0) / SS_ideal))
    else:
        mu_sr_ideal = 1000 / (1 + (E_eff_ideal * 1e-5)**2)
        mu_eff_bulk_ideal = 1 / (1/mu_ph + 1/mu_sr_ideal + 1/(1e16 / 1e10))
        mu_eff_ideal = max(mu_floor, mu_eff_bulk_ideal)
        
        I_long_ideal = 0.5 * mu_eff_ideal * C_ox * (W_cm / L_cm) * ((Vg - V_th0) ** 2)
        theta_sat_ideal = mu_eff_ideal * (Vg - V_th0) / (2 * v_sat * L_cm)
        Id_ideal.append((I_long_ideal / (1 + theta_sat_ideal)) + 1e-12)

# ==========================================
# 3. 통합 시각화 패널 구성
# ==========================================
fig = make_subplots(
    rows=1, cols=2, 
    horizontal_spacing=0.15,
    specs=[[{"type": "xy"}, {"secondary_y": True}]],
    subplot_titles=(f"소자 내부 시각화 (L = {L_nm} nm 기준)", "통합 전달 특성 (단채널 효과 및 하한선 반영)")
)

# --- [왼쪽] 소자 단면 (채널 길이 L_nm에 반응하여 동적 스케일링) ---
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
fig.add_annotation(x=L_nm/2, y=2.2, text="SiO₂ Interface", showarrow=False, font=dict(color="white", size=10), row=1, col=1)

np.random.seed(42) 
num_nit = int(Nit_slider * 2) 
if num_nit > 0:
    # HCI 트랩이므로 Drain 쪽에 치우치게 생성 (L_nm 기준)
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

# --- [오른쪽] 통합 I-V 및 이동도 (하한선 명시) ---
fig.add_trace(go.Scatter(x=Vg_sweep, y=Id_ideal, mode='lines', line=dict(color='gray', width=2, dash='dash'), name='I_d (Fresh)'), row=1, col=2, secondary_y=False)
fig.add_trace(go.Scatter(x=Vg_sweep, y=Id_degraded, mode='lines', line=dict(color='cyan', width=3), name='I_d (Degraded)'), row=1, col=2, secondary_y=False)
fig.add_trace(go.Scatter(x=Vg_sweep, y=mu_eff_array, mode='lines', line=dict(color='orange', width=4), name='유효 이동도 (μ_eff)'), row=1, col=2, secondary_y=True)

# 하한선 기준선 (가이드라인 추가)
fig.add_shape(type="line", x0=0, y0=mu_floor, x1=1.5, y1=mu_floor, line=dict(color="orange", width=1, dash="dot"), row=1, col=2, secondary_y=True)
fig.add_annotation(x=1.3, y=mu_floor+15, text=f"Mobility Floor ({mu_floor})", showarrow=False, font=dict(color="orange", size=10), row=1, col=2, secondary_y=True)

fig.update_xaxes(title_text="Gate Voltage (V_g) [V]", row=1, col=2)
fig.update_yaxes(title_text="Drain Current (I_d) [A]", type="log", range=[-12, -2], row=1, col=2, secondary_y=False)
fig.update_yaxes(title_text="Effective Mobility [cm²/V·s]", range=[0, 400], showgrid=False, row=1, col=2, secondary_y=True)

fig.update_layout(height=600, template="plotly_dark", legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5))
st.plotly_chart(fig, use_container_width=True)

# ==========================================
# 4. 소자 상태 파라미터 추출
# ==========================================
st.divider()
st.subheader("📊 실시간 파라미터 추출")
c1, c2, c3, c4 = st.columns(4)

c1.metric("문턱 전압 변동 (ΔV_th)", f"+{delta_Vth*1000:.1f} mV", "N_it & N_ot 통합 반영", delta_color="inverse")
c2.metric("Subthreshold Swing (SS)", f"{SS_degraded * 1000:.1f} mV/dec", f"+{(SS_degraded - SS_ideal)*1000:.1f} mV (열화)", delta_color="inverse")
c3.metric("최고 이동도 (Max μ_eff)", f"{max(mu_eff_array):.1f} cm²/V·s", f"하한선 {mu_floor} 도달 여부 관찰", delta_color="normal")
# 전류가 Linear Regime인지 Quadratic Regime인지 확인용 지표
theta_val = max(mu_eff_array) * (1.5 - V_th_degraded) / (2 * v_sat * L_cm) if 1.5 > V_th_degraded else 0
c4.metric("단채널 포화 계수 (θ)", f"{theta_val:.2f}", "1.0 이상이면 속도 포화 지배적", delta_color="off")
