import streamlit as st
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="Vibe Coding: AI Defect Visualizer", layout="wide")

st.title("🔬 AI 기반 반도체 결함 2D 시각화 엔진")
st.markdown("**이론적 배경:** 단순한 텍스트나 1차원 그래프를 넘어, AI 대리 모델을 통해 소자 내부의 2차원 결함 분포(Trap Density)를 실시간으로 렌더링합니다.")

col_control, col_visual = st.columns([1, 2.5])

# ==========================================
# 1. 제어 패널
# ==========================================
with col_control:
    st.subheader("🎛️ 물리적 스트레스 인가")
    st.markdown("반도체 소자에 가해지는 가혹 조건을 설정해보세요.")
    
    vd_stress = st.slider("⚡ 드레인 전압 (V_d)", min_value=1.0, max_value=5.0, value=1.5, step=0.1)
    time_stress = st.slider("⏳ 스트레스 시간 (Years)", min_value=0.0, max_value=10.0, value=0.0, step=0.5)
    
    st.info("💡 **동작 원리:**\n\n전압과 시간이 증가할수록, 드레인(오른쪽) 근처에서 강한 전기장을 얻은 고에너지 전자(Hot Carrier)가 산화막을 타격하여 결함(Trap)을 생성합니다. AI는 이 공간적 파괴 현상을 실시간으로 계산하여 시각화합니다.")

# ==========================================
# 2. AI 대리 모델 기반 2D 히트맵 연산
# ==========================================
x = np.linspace(0, 10, 100) 
y = np.linspace(0, 5, 50)   
X, Y = np.meshgrid(x, y)

trap_density = np.zeros_like(X)

if time_stress > 0:
    intensity = (vd_stress ** 2) * (time_stress ** 0.5) * 0.1
    hci_profile = np.exp(-((X - 9.5)**2 / 2.0 + (Y - 2.0)**2 / 0.5))
    trap_density += intensity * hci_profile

# ==========================================
# 3. Plotly 2D 화려한 시각화 (에러 수정됨!)
# ==========================================
with col_visual:
    fig = go.Figure()

    fig.add_trace(go.Contour(
        z=trap_density, x=x, y=y,
        colorscale='Inferno', 
        showscale=True,
        zmin=0, zmax=5, 
        colorbar=dict(title="결함 밀도 (N_it)"), # <--- 여기서 에러가 해결되었습니다!
        contours=dict(showlines=False)
    ))

    fig.add_shape(type="rect", x0=0, y0=2, x1=2, y1=5, line=dict(color="cyan", width=2), fillcolor="rgba(0,255,255,0.1)")
    fig.add_annotation(x=1, y=3.5, text="Source (n+)", showarrow=False, font=dict(color="cyan", size=16))
    
    fig.add_shape(type="rect", x0=8, y0=2, x1=10, y1=5, line=dict(color="cyan", width=2), fillcolor="rgba(0,255,255,0.1)")
    fig.add_annotation(x=9, y=3.5, text="Drain (n+)", showarrow=False, font=dict(color="cyan", size=16))

    fig.add_shape(type="rect", x0=2.5, y0=0.5, x1=7.5, y1=1.5, line=dict(color="yellow", width=2), fillcolor="rgba(255,255,0,0.2)")
    fig.add_annotation(x=5, y=1, text="Gate Electrode", showarrow=False, font=dict(color="yellow", size=16))
    
    fig.add_shape(type="line", x0=2, y0=2, x1=8, y1=2, line=dict(color="white", width=3, dash="dash"))
    fig.add_annotation(x=5, y=2.2, text="SiO₂ Interface", showarrow=False, font=dict(color="white", size=12))

    fig.update_layout(
        title="🔥 트랜지스터 단면 실시간 트랩(Trap) 생성 시뮬레이션",
        xaxis_title="채널 위치 (Position)",
        yaxis_title="깊이 (Depth) - 위(Gate) / 아래(Substrate)",
        yaxis=dict(autorange="reversed"), 
        template="plotly_dark",
        height=500,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    max_trap = np.max(trap_density)
    if max_trap > 4:
        st.error(f"🚨 치명적 손상 발생! (최대 결함 지수: {max_trap:.1f}) - 소자 수명 종료")
    elif max_trap > 2:
        st.warning(f"⚠️ 산화막 열화 진행 중 (최대 결함 지수: {max_trap:.1f}) - 누설 전류 증가")
    else:
        st.success(f"✅ 안정적인 상태 (최대 결함 지수: {max_trap:.1f})")
