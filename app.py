import streamlit as st
import torch
import torch.nn as nn
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# ==========================================
# 1. 모델 아키텍처 정의 (학습 코드와 동일해야 함)
# ==========================================
class VibeEngine(nn.Module):
    def __init__(self):
        super(VibeEngine, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )
    def forward(self, x):
        return self.network(x)

# ==========================================
# 2. AI 모델 및 스케일러 로드 (캐싱하여 속도 최적화)
# ==========================================
@st.cache_resource # Streamlit이 모델을 한 번만 메모리에 올리도록 캐싱
def load_ai_engine():
    model = VibeEngine()
    model.load_state_dict(torch.load('vibe_engine_model.pt'))
    model.eval() # 추론 모드로 전환
    scaler_X = joblib.load('scaler_X.pkl')
    scaler_Y = joblib.load('scaler_Y.pkl')
    return model, scaler_X, scaler_Y

model, scaler_X, scaler_Y = load_ai_engine()

# ==========================================
# 3. Streamlit 프론트엔드 UI 구성
# ==========================================
st.set_page_config(page_title="Defect AI Simulator", layout="wide")

st.title("⚡ AI 기반 반도체 결함 & 수명 시뮬레이터")
st.markdown("**정적인 물리 공식을 넘어, 실시간으로 변화하는 소자의 수명을 예측합니다.**")

# 사이드바 (사용자 입력 컨트롤)
st.sidebar.header("🎛️ 스트레스 조건 설정")
stress_voltage = st.sidebar.slider("Gate Voltage [V_g]", min_value=0.5, max_value=3.3, value=1.5, step=0.1)
max_time = st.sidebar.slider("Simulation Time [Seconds]", min_value=1000, max_value=10000, value=5000, step=500)

# ==========================================
# 4. 실시간 AI 추론 및 데이터 생성
# ==========================================
# 1초부터 사용자가 설정한 최대 시간까지 100개의 포인트 생성
time_array = np.linspace(1, max_time, 100)
voltage_array = np.full_like(time_array, stress_voltage)

# 모델에 넣기 위해 형태 맞추기 (Voltage, Time)
input_data = np.column_stack((voltage_array, time_array))

# 스케일링 적용
input_scaled = scaler_X.transform(input_data)
input_tensor = torch.tensor(input_scaled, dtype=torch.float32)

# AI 모델 추론 (단 0.01초 만에 물리적 결과 연산 완료!)
with torch.no_grad():
    output_scaled = model(input_tensor).numpy()

# 스케일링 원복 (실제 물리량으로 변환)
output_data = scaler_Y.inverse_transform(output_scaled)
delta_vth = output_data[:, 0]
n_it = output_data[:, 1]

# ==========================================
# 5. Plotly를 활용한 기깔나는 동적 시각화
# ==========================================
col1, col2 = st.columns(2)

with col1:
    st.subheader("📈 문턱 전압 변동 (ΔVth) 예측")
    fig_vth = go.Figure()
    fig_vth.add_trace(go.Scatter(x=time_array, y=delta_vth, mode='lines', 
                                 line=dict(color='firebrick', width=3),
                                 name='ΔVth Shift'))
    fig_vth.update_layout(xaxis_title="Stress Time (s)", yaxis_title="ΔVth (V)", 
                          template="plotly_dark", margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig_vth, use_container_width=True)

with col2:
    st.subheader("⚛️ 계면 트랩 밀도 (N_it) 증가")
    fig_nit = go.Figure()
    fig_nit.add_trace(go.Scatter(x=time_array, y=n_it, mode='lines', fill='tozeroy',
                                 line=dict(color='royalblue', width=3),
                                 name='Trap Density'))
    fig_nit.update_layout(xaxis_title="Stress Time (s)", yaxis_title="N_it (cm⁻²)", 
                          template="plotly_dark", margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig_nit, use_container_width=True)

# 하단 요약 매트릭스
st.divider()
st.metric(label=f"최종 {max_time}초 후 예상 ΔVth", value=f"{delta_vth[-1]:.4f} V", delta="Critical!" if delta_vth[-1] > 0.1 else "Stable", delta_color="inverse")
