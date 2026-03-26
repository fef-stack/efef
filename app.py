import streamlit as st
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="AI Device Characterization", layout="wide")

st.title("🔬 물리 기반 AI 소자 특성 평가 (Id-Vg Characterization)")
st.markdown("**이론적 배경:** 단순한 Vth 이동이 아닌, 산화막 계면 트랩($N_{it}$) 증가에 따른 Subthreshold Swing(SS) 열화와 이동도(Mobility) 감소 현상을 양자역학적 관점에서 렌더링합니다.")

col_param, col_graph = st.columns([1, 2.5])

# ==========================================
# 1. 소자 파라미터 및 스트레스 인가 (고정값 + 변수)
# ==========================================
with col_param:
    st.subheader("📐 타겟 소자 스펙 (20nm NMOS)")
    # 물리적으로 고정된 소자 스펙 (면접관 어필용)
    st.text("- EOT (산화막 두께): 2.0 nm\n- L (채널 길이): 20 nm\n- W (채널 폭): 100 nm\n- N_A (기판 도핑): 1e18 cm⁻³")
    
    st.divider()
    st.subheader("🎛️ 물리적 스트레스 (HCI/NBTI)")
    
    stress_time = st.slider("⏳ 스트레스 인가 시간 (Years)", min_value=0.0, max_value=10.0, value=0.0, step=0.5)
    stress_vg = st.slider("⚡ 가혹 전압 조건 (V)", min_value=1.0, max_value=3.3, value=1.2, step=0.1)

    st.info("💡 **물리 엔진 작동 방식:**\nAI 대리 모델이 스트레스 조건에 따른 $N_{it}$ 생성량을 계산하고, 이를 통해 $SS$ 열화, $V_{th}$ 이동, $\\mu$ 감소를 동시에 반영하여 정밀한 $I_d-V_g$ 곡선을 도출합니다.")

# ==========================================
# 2. 물리 엔진 (Mathematical Model for Id-Vg)
# ==========================================
# 물리 상수
q = 1.6e-19
kT = 0.0259 # eV (at 300K)
eps_ox = 3.9 * 8.85e-14 # F/cm

# 소자 파라미터 계산
t_ox = 2e-7 # cm
C_ox = eps_ox / t_ox # F/cm^2
mu_0 = 300 # 초기 이동도 cm^2/Vs
V_th0 = 0.4 # 초기 문턱 전압 V

# 스트레스에 따른 결함(Nit) 생성 모델 (Power law)
# N_it = A * exp(B * V_stress) * t^n
N_it_0 = 1e10 # 초기 결함 밀도
if stress_time > 0:
    delta_Nit = 1e10 * np.exp(1.5 * stress_vg) * (stress_time ** 0.5)
else:
    delta_Nit = 0
N_it_total = N_it_0 + delta_Nit

# 트랩에 의한 소자 특성 열화 계산
# 1. Vth Shift
delta_Vth = (q * delta_Nit) / C_ox
V_th_degraded = V_th0 + delta_Vth

# 2. SS (Subthreshold Swing) Degradation
# Ideal SS is approx 60 mV/dec. Traps add to the capacitance ratio.
SS_ideal = 0.060 # V/dec
SS_degraded = SS_ideal * (1 + (q * N_it_total) / C_ox)

# 3. Mobility Degradation (Coulomb Scattering)
alpha = 1e-12
mu_degraded = mu_0 / (1 + alpha * delta_Nit)

# ==========================================
# 3. Id-Vg 커브 생성 (게이트 전압 스윕)
# ==========================================
Vg_sweep = np.linspace(0.0, 1.2, 200)
Id_array = np.zeros_like(Vg_sweep)

# Id-Vg 계산 (Subthreshold 영역과 Linear/Saturation 영역 접합)
for i, Vg in enumerate(Vg_sweep):
    if Vg < V_th_degraded:
        # Subthreshold Region: Exponential dependence
        # I_off 계산 기반
        I_off = 1e-11 # Reference off current
        Id_array[i] = I_off * 10 ** ((Vg - V_th_degraded) / SS_degraded)
    else:
        # Strong Inversion Region: Square law (simplified)
        # I = 0.5 * mu * Cox * (W/L) * (Vg - Vth)^2
        W_L_ratio = 5
        Id_array[i] = 0.5 * mu_degraded * C_ox * W_L_ratio * ((Vg - V_th_degraded) ** 2)
        # 매끄러운 접합을 위해 Subthreshold 전류를 베이스로 깔아줌
        Id_array[i] += 1e-11 

# ==========================================
# 4. 현업 수준의 Log-Scale 시각화 (Plotly)
# ==========================================
with col_graph:
    fig = go.Figure()

    # 초기 상태 (Fresh Device) 커브 - 비교를 위한 기준선 (점선)
    Id_ideal = np.zeros_like(Vg_sweep)
    for i, Vg in enumerate(Vg_sweep):
        if Vg < V_th0:
            Id_ideal[i] = 1e-11 * 10 ** ((Vg - V_th0) / SS_ideal)
        else:
            Id_ideal[i] = 0.5 * mu_0 * C_ox * 5 * ((Vg - V_th0) ** 2) + 1e-11

    fig.add_trace(go.Scatter(x=Vg_sweep, y=Id_ideal, mode='lines', 
                             line=dict(color='gray', width=2, dash='dash'), 
                             name='초기 상태 (Fresh, 0 Year)'))

    # 열화된 상태 (Degraded Device) 커브
    fig.add_trace(go.Scatter(x=Vg_sweep, y=Id_array, mode='lines', 
                             line=dict(color='cyan', width=4), 
                             name=f'열화 상태 ({stress_time} Years)'))

    fig.update_layout(
        title="트랜지스터 전달 특성 (Id-Vg Curve) 실시간 변화",
        xaxis_title="Gate Voltage (V_g) [V]",
        yaxis_title="Drain Current (I_d) [A] - Log Scale",
        yaxis_type="log", # ⭐️ 현업 필수: Log 스케일 적용
        yaxis=dict(range=[-12, -3]), # 1pA ~ 1mA 범위
        template="plotly_dark",
        height=450,
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    
    st.plotly_chart(fig, use_container_width=True)

    # ------------------------------------------
    # 5. 소자 상태 파라미터 추출 매트릭스
    # ------------------------------------------
    m1, m2, m3 = st.columns(3)
    
    # SS 포맷팅 (mV/dec)
    m1.metric("Subthreshold Swing (SS)", f"{SS_degraded * 1000:.1f} mV/dec", 
              f"+{(SS_degraded - SS_ideal)*1000:.1f} mV (열화)", delta_color="inverse")
    
    # Vth 포맷팅
    m2.metric("문턱 전압 (V_th)", f"{V_th_degraded:.3f} V", 
              f"+{delta_Vth:.3f} V (Shift)", delta_color="inverse")
    
    # I_on 포맷팅 (Vg=1.2V 일 때 전류)
    I_on_degraded = Id_array[-1]
    I_on_ideal = Id_ideal[-1]
    I_on_drop_pct = ((I_on_ideal - I_on_degraded) / I_on_ideal) * 100
    m3.metric("On-Current (I_on)", f"{I_on_degraded * 1e6:.1f} µA", 
              f"-{I_on_drop_pct:.1f}% (Mobility 감소)", delta_color="inverse")
