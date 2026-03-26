import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Vibe Coding: AI Device Simulator", layout="wide")

st.title("🔬 통합 물리 시뮬레이터: 소자 내부 현상 및 전달 특성 열화")
st.markdown("---")
st.markdown("**이론적 배경:** 단순한 그래프 이동을 넘어, 고전압 스트레스에 의한 **HCI(Hot Carrier Injection) 현상**을 미시적으로 시각화하고, 이로 인한 **Subthreshold Swing(SS) 열화와 이동도(Mobility) 감소**가 엄밀한 $I_d-V_g$ 곡선에 어떻게 반영되는지 실시간으로 증명합니다.")

# ==========================================
# 0. 제어 패널 (사이드바)
# ==========================================
st.sidebar.header("🎛️ 물리적 스트레스 인가")
# 타겟 소자 스펙 고정 (면접관 어필용)
st.sidebar.markdown("### 📐 타겟 소자: 20nm급 NMOS\n- EOT: 2.0 nm, L: 20 nm, W: 100 nm")
st.sidebar.divider()

stress_time = st.sidebar.slider("⏳ 스트레스 시간 (Years)", min_value=0.0, max_value=10.0, value=0.0, step=0.5)
stress_vd = st.sidebar.slider("⚡ 드레인 가혹 전압 (V_d,stress)", min_value=1.0, max_value=3.3, value=1.5, step=0.1)

st.sidebar.info("💡 **동작 원리:**\n\nAI 엔진이 입력된 스트레스 조건에 따른 계면 트랩($N_{it}$) 생성량을 드레인 근처 공간 분포로 계산합니다. 이 $N_{it}$ 값은 동시에 물리 법칙에 입력되어 $I_d-V_g$ 커브를 실시간으로 열화시킵니다.")

# ==========================================
# 1. 물리 엔진 (Mathematical Model & AI Proxy)
# ==========================================
# 물리 상수 및 소자 파라미터
q = 1.6e-19
kT = 0.0259 # eV (at 300K)
eps_ox = 3.9 * 8.85e-14 # F/cm
t_ox = 2e-7 # cm
C_ox = eps_ox / t_ox # F/cm^2
mu_0 = 300 # 초기 이동도
V_th0 = 0.4 # 초기 문턱 전압
SS_ideal = 0.060 # Ideal SS V/dec

# A. 미시적 결함(Nit) 생성 및 공간 분포 모델링 (AI Proxy)
delta_Nit_max = 0
trap_distribution = None

# MOSFET 단면 그리드 (2D Visualization용)
x_visual = np.linspace(0, 10, 100) # 채널 길이
y_visual = np.linspace(0, 5, 50)   # 깊이
X, Y = np.meshgrid(x_visual, y_visual)

if stress_time > 0:
    # 스트레스에 따른 결함 밀도 최대값 (Power law 기반)
    # N_it ∝ exp(Vd) * t^n
    delta_Nit_max = 1e11 * np.exp(1.2 * stress_vd) * (stress_time ** 0.5)
    
    # HCI 현상 특성상 드레인 쪽(x=9.5), 계면(y=2) 근처에 가우시안 분포로 트랩 생성
    trap_profile = np.exp(-((X - 9.5)**2 / 2.0 + (Y - 2.0)**2 / 0.5))
    trap_distribution = delta_Nit_max * trap_profile
else:
    trap_distribution = np.zeros_like(X)

# B. 거시적 소자 특성 열화 계산 (for Id-Vg Curve)
# 시뮬레이션에 사용할 평균 Nit 값 (간략화)
avg_delta_Nit = delta_Nit_max * 0.1 # 분포의 최대값 일부를 평균으로 사용
N_it_total = 1e10 + avg_delta_Nit # 초기 1e10 기반

# 1. Vth Shift
delta_Vth = (q * avg_delta_Nit) / C_ox
V_th_degraded = V_th0 + delta_Vth

# 2. SS Degradation
SS_degraded = SS_ideal * (1 + (q * N_it_total) / C_ox)

# 3. Mobility Degradation (Coulomb Scattering)
alpha = 1e-12
mu_degraded = mu_0 / (1 + alpha * avg_delta_Nit)

# C. Id-Vg 커브 데이터 생성 (게이트 전압 스윕)
Vg_sweep = np.linspace(0.0, 1.2, 200)
Id_ideal = np.zeros_like(Vg_sweep)
Id_degraded = np.zeros_like(Vg_sweep)

for i, Vg in enumerate(Vg_sweep):
    # Ideal Curve
    if Vg < V_th0:
        Id_ideal[i] = 1e-11 * 10 ** ((Vg - V_th0) / SS_ideal)
    else:
        Id_ideal[i] = 0.5 * mu_0 * C_ox * 5 * ((Vg - V_th0) ** 2) + 1e-11
    
    # Degraded Curve
    if Vg < V_th_degraded:
        Id_degraded[i] = 1e-11 * 10 ** ((Vg - V_th_degraded) / SS_degraded)
    else:
        Id_degraded[i] = 0.5 * mu_degraded * C_ox * 5 * ((Vg - V_th_degraded) ** 2) + 1e-11

# ==========================================
# 2. 통합 시각화 (subplot 활용)
# ==========================================
# 2개의 서브플롯 생성 (왼쪽: 소자 시각화, 오른쪽: Id-Vg)
fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.1,
                    subplot_titles=("소자 내부 현상 시각화 (HCI 결함 히트맵)", 
                                    "전달 특성 (Id-Vg) 곡선 변화"))

# --- [왼쪽] 소자 내부 현상 시각화 ---
# 히트맵 (결함 분포)
fig.add_trace(go.Contour(
    z=trap_distribution, x=x_visual, y=y_visual,
    colorscale='Inferno', 
    showscale=True,
    colorbar=dict(title="결함 밀도 (N_it)", x=0.45, len=0.75),
    contours=dict(showlines=False),
    zmin=0, zmax=5e12 # 최대 결함 범위 고정
), row=1, col=1)

# 구조물 테두리 및 라벨
# Source/Drain
fig.add_shape(type="rect", x0=0, y0=2, x1=2, y1=5, line=dict(color="cyan", width=2), fillcolor="rgba(0,255,255,0.05)", row=1, col=1)
fig.add_annotation(x=1, y=3.5, text="Source (n+)", showarrow=False, font=dict(color="cyan", size=14), row=1, col=1)
fig.add_shape(type="rect", x0=8, y0=2, x1=10, y1=5, line=dict(color="cyan", width=2), fillcolor="rgba(0,255,255,0.05)", row=1, col=1)
fig.add_annotation(x=9, y=3.5, text="Drain (n+)", showarrow=False, font=dict(color="cyan", size=14), row=1, col=1)
# Gate/Interface
fig.add_shape(type="rect", x0=2.5, y0=0.5, x1=7.5, y1=1.5, line=dict(color="yellow", width=2), fillcolor="rgba(255,255,0,0.1)", row=1, col=1)
fig.add_annotation(x=5, y=1, text="Gate", showarrow=False, font=dict(color="yellow", size=14), row=1, col=1)
fig.add_shape(type="line", x0=2, y0=2, x1=8, y1=2, line=dict(color="white", width=2, dash="dash"), row=1, col=1)
fig.add_annotation(x=5, y=2.2, text="SiO₂ Interface", showarrow=False, font=dict(color="white", size=10), row=1, col=1)

fig.update_xaxes(title_text="채널 위치 (Position)", row=1, col=1)
fig.update_yaxes(title_text="깊이 (Depth)", autorange="reversed", row=1, col=1)

# --- [오른쪽] 전달 특성 (Id-Vg) 곡선 변화 ---
fig.add_trace(go.Scatter(x=Vg_sweep, y=Id_ideal, mode='lines', 
                         line=dict(color='gray', width=2, dash='dash'), 
                         name='초기 상태 (Fresh)'), row=1, col=2)

fig.add_trace(go.Scatter(x=Vg_sweep, y=Id_degraded, mode='lines', 
                         line=dict(color='cyan', width=4), 
                         name=f'열화 상태 ({stress_time}년)'), row=1, col=2)

fig.update_xaxes(title_text="Gate Voltage (V_g) [V]", row=1, col=2)
fig.update_yaxes(title_text="Drain Current (I_d) [A] - Log Scale", type="log", range=[-12, -3], row=1, col=2)

# 전체 레이아웃 설정
fig.update_layout(
    height=550,
    template="plotly_dark",
    margin=dict(l=20, r=20, t=60, b=20),
    legend=dict(yanchor="bottom", y=0.01, xanchor="right", x=0.99)
)

st.plotly_chart(fig, use_container_width=True)

# ==========================================
# 3. 소자 상태 파라미터 추출 매트릭스 (Bottom)
# ==========================================
st.divider()
st.subheader("📊 추출된 물리 파라미터 변화")
m1, m2, m3 = st.columns(3)

# SS 포맷팅 (mV/dec)
m1.metric("Subthreshold Swing (SS)", f"{SS_degraded * 1000:.1f} mV/dec", 
          f"+{(SS_degraded - SS_ideal)*1000:.1f} mV (열화)", delta_color="inverse")
# Vth 포맷팅
m2.metric("문턱 전압 (V_th)", f"{V_th_degraded:.3f} V", 
          f"+{delta_Vth:.3f} V (Shift)", delta_color="inverse")
# I_on 포맷팅 (Vg=1.2V 일 때 전류)
I_on_drop_pct = ((Id_ideal[-1] - Id_degraded[-1]) / Id_ideal[-1]) * 100
m3.metric("On-Current (I_on @1.2V)", f"{Id_degraded[-1] * 1e6:.1f} µA", 
          f"-{I_on_drop_pct:.1f}% (Mobility 감소)", delta_color="inverse")
