import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="고도화된 통합 물리 시뮬레이터", layout="wide")

st.title("🔬 통합 소자 물리 시뮬레이터 (Short-Channel Effect & Reliability)")
st.markdown("본 시뮬레이터는 전공 입문자가 반도체 소자의 단채널 효과(SCE)와 열화 경향(HCI, BTI)을 직관적으로 탐색할 수 있도록 설계된 **정성적 교육용 대리 모델(Qualitative Educational Surrogate Model)**입니다.")
st.markdown("---")

# ==========================================
# 1. 제어 패널 (사이드바)
# ==========================================
st.sidebar.header("📐 소자 구조 및 측정 환경 (Measurement)")
L_nm = st.sidebar.slider("채널 길이 (L) [nm]", min_value=10, max_value=200, value=20, step=5, 
                         help="L을 30nm 이하로 줄이면 Gate 통제력이 상실되며 누설 전류가 증가합니다.")
T_K = st.sidebar.slider("🌡️ 동작 온도 (T) [K]", 300, 400, 300, 10)
N_A_str = st.sidebar.select_slider("🧬 채널 도핑 농도 (N_A) [cm⁻³]", options=["1e16", "5e16", "1e17", "5e17", "1e18"], value="1e17")
N_A = float(N_A_str)
V_d_read = st.sidebar.slider("측정 드레인 전압 (V_d,read) [V]", 0.1, 2.0, 1.0, 0.1, help="I-V 곡선 측정 시 가해지는 드레인 전압 (DIBL에 영향)")

st.sidebar.divider()
st.sidebar.header("⚡ 물리적 효과 제어 (Attribution)")
apply_sce = st.sidebar.checkbox("단채널 효과 (SCE) 적용", value=True, help="체크 시 Vth Roll-off 및 DIBL 효과가 활성화됩니다.")
apply_hci = st.sidebar.checkbox("HCI 열화 모델 적용", value=True)
apply_bti = st.sidebar.checkbox("BTI 열화 모델 적용", value=True)

st.sidebar.divider()
st.sidebar.header("🔥 스트레스 인가 (Stress Phase)")
stress_time = st.sidebar.slider("⏳ 스트레스 시간 (Years)", 0.0, 10.0, 0.0, 0.5)
stress_vd = st.sidebar.slider("⚡ 가혹 전압 (V_d,stress) [V]", 1.0, 3.3, 2.0, 0.1, help="열화를 유발하는 스트레스 전압")

# 물리 상수
q = 1.6e-19
k_B = 1.38e-23
k_eV = 8.617e-5

# 스트레스 모델에 의한 트랩 계산 (독립적 기여도 분리)
hci_trap = 0
bti_trap = 0

if stress_time > 0:
    if apply_hci:
        electric_field_factor = (N_A / 1e17) ** 0.5 
        hci_temp_factor = (300 / T_K) ** 1.5 
        hci_trap = 2e10 * np.exp(1.0 * stress_vd) * (stress_time ** 0.5) * electric_field_factor * hci_temp_factor
    
    if apply_bti:
        E_a = 0.15 
        bti_temp_factor = np.exp(-(E_a / k_eV) * (1/T_K - 1/300))
        bti_trap = 3e10 * np.exp(1.2 * stress_vd) * (stress_time ** 0.5) * bti_temp_factor

delta_trap_base = hci_trap + bti_trap

st.sidebar.markdown("### 🪤 개별 트랩 수동 추가 (Manual Injection)")
Nit_slider = st.sidebar.slider("추가 계면 트랩 (ΔN_it) [x 10¹¹ cm⁻²]", 0.0, 50.0, 0.0, 0.1)
Not_slider = st.sidebar.slider("추가 산화막 트랩 (ΔN_ot) [x 10¹¹ cm⁻²]", 0.0, 50.0, 0.0, 0.1)

# Baseline Trap 설정 및 추가 트랩 계산
baseline_Nit = 1e10
delta_Nit_total = (hci_trap + bti_trap) * 0.8 + (Nit_slider * 1e11)
delta_Not_total = (hci_trap + bti_trap) * 0.2 + (Not_slider * 1e11)
N_it_current = baseline_Nit + delta_Nit_total
N_ot_current = delta_Not_total

# ==========================================
# 2. 물리 엔진 연산
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

# 이상적(Long Channel 기준) 물리량
phi_F = (k_B * T_K / q) * np.log(N_A / n_i)
Q_dep = np.sqrt(2 * q * eps_si * N_A * (2 * phi_F)) 
W_dep = Q_dep / (q * N_A)
C_d = eps_si / W_dep

# Baseline(Fresh) 상태 연산
V_th0_long = 0.4 + (2 * phi_F) + (Q_dep / C_ox)
SS_ideal_long = np.log(10) * (k_B * T_K / q) * (1 + (C_d + q * baseline_Nit) / C_ox)

# SCE 적용 여부에 따른 모델링 (V_d_read 사용)
lambda_char = 15.0 # nm
sce_factor = np.exp(-L_nm / lambda_char) if apply_sce else 0.0

V_th_rolloff = 0.4 * sce_factor 
DIBL_shift = 0.1 * V_d_read * sce_factor 
V_th_fresh = V_th0_long - V_th_rolloff - DIBL_shift
SS_fresh = SS_ideal_long * (1 + 4.0 * sce_factor)

# 트랩 열화 반영 (Degraded 상태)
delta_Vth = (q * (delta_Nit_total + delta_Not_total)) / C_ox
V_th_degraded = V_th_fresh + delta_Vth
SS_degraded = SS_fresh * (1 + (C_d + q * N_it_current) / C_ox) / (1 + (C_d + q * baseline_Nit) / C_ox)

# I-V 및 Mobility 커브 연산
Vg_sweep = np.linspace(0.0, 4.0, 400)
Id_fresh, Id_degraded, mu_eff_array = [], [], []
I_th = 1e-7 * (W_cm / L_cm)

for Vg in Vg_sweep:
    # --- [Degraded 커브 계산] ---
    Q_inv_degraded = C_ox * max(0, Vg - V_th_degraded)
    E_eff_degraded = (Q_dep + 0.5 * Q_inv_degraded) / eps_si

    mu_ph = 300 * ((300 / T_K) ** 1.5)
    mu_sr = 1000 / (1 + (E_eff_degraded * 1e-5)**2) 
    mu_coulomb = 1e16 / max(1e10, N_it_current + N_ot_current)
    
    mu_eff_bulk = 1 / (1/mu_ph + 1/mu_sr + 1/mu_coulomb)
    mu_eff = max(mu_floor, mu_eff_bulk) 
    mu_eff_array.append(mu_eff)
    
    if Vg < V_th_degraded:
        I_sub = I_th * 10 ** ((Vg - V_th_degraded) / SS_degraded)
        Id_degraded.append(I_sub)
    else:
        I_long = 0.5 * mu_eff * C_ox * (W_cm / L_cm) * ((Vg - V_th_degraded) ** 2)
        theta_sat = mu_eff * (Vg - V_th_degraded) / (2 * v_sat * L_cm)
        Id_degraded.append((I_long / (1 + theta_sat)) + I_th)

    # --- [Fresh 커브 계산] ---
    Q_inv_fresh = C_ox * max(0, Vg - V_th_fresh)
    E_eff_fresh = (Q_dep + 0.5 * Q_inv_fresh) / eps_si
    
    mu_sr_fresh = 1000 / (1 + (E_eff_fresh * 1e-5)**2)
    mu_coulomb_fresh = 1e16 / baseline_Nit
    mu_eff_bulk_fresh = 1 / (1/mu_ph + 1/mu_sr_fresh + 1/mu_coulomb_fresh)
    mu_eff_fresh = max(mu_floor, mu_eff_bulk_fresh)
    
    if Vg < V_th_fresh:
        I_sub_fresh = I_th * 10 ** ((Vg - V_th_fresh) / SS_fresh)
        Id_fresh.append(I_sub_fresh)
    else:
        I_long_fresh = 0.5 * mu_eff_fresh * C_ox * (W_cm / L_cm) * ((Vg - V_th_fresh) ** 2)
        theta_sat_fresh = mu_eff_fresh * (Vg - V_th_fresh) / (2 * v_sat * L_cm)
        Id_fresh.append((I_long_fresh / (1 + theta_sat_fresh)) + I_th)

# 측정 지표 추출 (Vg=1.5V 기준)
idx_read = np.argmin(np.abs(Vg_sweep - 1.5))
I_on = Id_degraded[idx_read]
I_off = Id_degraded[0] # Vg=0
Ion_Ioff_ratio = I_on / I_off

# ==========================================
# 3. 통합 시각화 패널 구성
# ==========================================
fig = make_subplots(
    rows=1, cols=2, 
    horizontal_spacing=0.15,
    specs=[[{"type": "xy"}, {"secondary_y": True}]],
    subplot_titles=(f"소자 내부 2D Trap Marker Visualization (L = {L_nm} nm)", "통합 전달 특성 (단채널 누설 붕괴 시각화)")
)

# [왼쪽] 소자 단면 (2D Trap Marker Visualization)
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
num_nit = int((delta_Nit_total/1e11) * 2) 
if num_nit > 0:
    x_nit = np.clip(np.random.normal(L_nm*0.7, L_nm*0.1, num_nit), G_start, G_end)
    y_nit = np.random.normal(2.0, 0.05, num_nit)
    fig.add_trace(go.Scatter(x=x_nit, y=y_nit, mode='markers', marker=dict(color='cyan', size=5, opacity=0.8), name='N_it (Interface)'), row=1, col=1)

num_not = int((delta_Not_total/1e11) * 2)
if num_not > 0:
    x_not = np.random.uniform(G_start, G_end, num_not)
    y_not = np.random.uniform(1.6, 1.9, num_not)
    fig.add_trace(go.Scatter(x=x_not, y=y_not, mode='markers', marker=dict(color='red', size=6, opacity=0.8), name='N_ot (Oxide)'), row=1, col=1)

fig.update_xaxes(title_text="Channel Position (nm)", range=[0, L_nm], row=1, col=1)
fig.update_yaxes(title_text="Depth (nm)", range=[5, 0], row=1, col=1)

# [오른쪽] 통합 I-V 및 이동도
fig.add_trace(go.Scatter(x=Vg_sweep, y=Id_fresh, mode='lines', line=dict(color='gray', width=2, dash='dash'), name='I_d (Fresh)'), row=1, col=2, secondary_y=False)
fig.add_trace(go.Scatter(x=Vg_sweep, y=Id_degraded, mode='lines', line=dict(color='cyan', width=3), name='I_d (Degraded)'), row=1, col=2, secondary_y=False)
fig.add_trace(go.Scatter(x=Vg_sweep, y=mu_eff_array, mode='lines', line=dict(color='orange', width=4), name='유효 이동도 (μ_eff)'), row=1, col=2, secondary_y=True)

fig.update_xaxes(title_text="Gate Voltage (V_g) [V]", row=1, col=2)
fig.update_yaxes(title_text="Drain Current (I_d) [A]", type="log", range=[-12, -2], row=1, col=2, secondary_y=False)
fig.update_yaxes(title_text="Effective Mobility [cm²/V·s]", range=[0, 400], showgrid=False, row=1, col=2, secondary_y=True)

fig.update_layout(height=600, template="plotly_dark", legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5))
st.plotly_chart(fig, use_container_width=True)

# ==========================================
# 4. 소자 상태 파라미터 및 AI 튜터링 (Interactive)
# ==========================================
st.divider()
st.subheader("📊 실시간 소자 성능 파라미터")
c1, c2, c3, c4 = st.columns(4)

c1.metric("초기 문턱 전압 (V_th,fresh)", f"{V_th_fresh:.3f} V", f"Roll-off & DIBL 반영", delta_color="off")
c2.metric("Subthreshold Swing (SS)", f"{SS_degraded * 1000:.1f} mV/dec", f"현재 붕괴도 (이상치: ~60)", delta_color="inverse")
c3.metric("Off-Current (누설 전류)", f"{I_off:.1e} A", f"ΔVth = {delta_Vth:.3f}V Shift", delta_color="inverse")

ratio_str = f"10^{np.log10(Ion_Ioff_ratio):.1f}"
c4.metric("I_on / I_off 점멸비", ratio_str, "측정 기준: Vg=1.5V, Vd=1.0V", delta_color="normal" if Ion_Ioff_ratio > 1e4 else "inverse")

# --- AI 물리 튜터링 (학습자 맞춤형 자동 해설) ---
st.subheader("🤖 AI 물리 튜터링 (Interactive Analysis)")
with st.container(border=True):
    # 1. SCE (Short Channel Effect) 상태 해석
    if apply_sce and L_nm <= 30:
        st.warning(f"**📉 단채널 효과(SCE) 경고:** 현재 채널 길이({L_nm}nm)가 짧아 게이트가 채널 전위를 완벽히 통제하지 못하고 있습니다. 이로 인해 **Vth Roll-off** 현상이 발생하여 초기 문턱전압이 현저히 낮아졌으며, I-V 곡선이 왼쪽으로 이동(누설 증가)했습니다. 슬라이더에서 채널 길이를 50nm 이상으로 늘려 변화를 관찰해 보세요.")
    elif apply_sce and L_nm > 30:
        st.success(f"**✅ Long Channel 거동:** 현재 채널 길이({L_nm}nm)에서는 게이트의 통제력이 안정적으로 유지되어 뚜렷한 단채널 효과(SCE)가 나타나지 않습니다.")

    # 2. 열화 (Degradation) 상태 해석
    if stress_time > 0:
        dominant_effect = "HCI" if hci_trap > bti_trap else "BTI"
        st.info(f"**⚡ 스트레스 열화 관찰 (Dominant: {dominant_effect}):** {stress_time}년 간의 스트레스(Vd={stress_vd}V)로 인해 계면 및 산화막 내부에 트랩이 생성되었습니다. 생성된 트랩 전하가 문턱전압을 양의 방향으로 이동(Positive Shift)시켰으며, 트랩에 의한 쿨롱 산란(Coulomb Scattering)으로 인해 캐리어의 이동도(μ_eff)가 감소하고 구동 전류(I_on)가 저하되는 현상을 확인할 수 있습니다.")
    else:
        st.info("**💡 팁:** 왼쪽 패널에서 '스트레스 시간'을 증가시켜 시간 경과에 따른 소자의 노화(Degradation) 과정을 관찰해 보세요.")

    # 3. On/Off Ratio (점멸비) 해석
    if Ion_Ioff_ratio < 1e4:
        st.error(f"**🚫 스위칭 특성 붕괴:** 현재 소자의 점멸비가 {ratio_str} 수준으로 떨어져 디지털 스위치로서의 기능을 상실했습니다. 누설 전류(I_off)가 너무 높거나 문턱전압이 심각하게 변동한 상태입니다.")
    else:
        st.write(f"- **점멸비 양호:** 현재 점멸비({ratio_str})는 디지털 논리 소자로서 작동 가능한 수준을 유지하고 있습니다.")

# ==========================================
# 5. Reference & Documentations
# ==========================================
st.divider()
with st.expander("📚 Reference & Model Validation (물리적 근거)"):
    st.markdown("""
    본 시뮬레이터는 다음 문헌의 물리적 수식 및 경험적 대리 모델(Surrogate model)을 기반으로 작성되었습니다.
    * **HCI (Hot Carrier Injection) Model:** 전계 및 온도에 따른 트랩 생성 (기반 문헌: *Takeda et al., "Empirical Model for Device Degradation Due to Hot-Carrier Injection", IEEE EDL, 1983*)
    * **BTI (Bias Temperature Instability):** Arrhenius 온도 의존성을 갖는 트랩 활성화 모델 (기반 문헌: *Alam et al., "A comprehensive model for PMOS NBTI degradation", Microelectronics Reliability, 2007*)
    * **Mobility Scattering:** Phonon, Surface Roughness, Coulomb 산란을 결합한 Matthiessen's Rule 적용
    * **SCE & DIBL:** 특성 길이($\lambda$) 기반의 지수 함수적 문턱 전압 강하 모델 적용
    """)
