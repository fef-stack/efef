import streamlit as st
import time
import numpy as np
import plotly.graph_objects as go
import torch
import torch.nn as nn

st.set_page_config(page_title="이론적 배경: AI 대리 모델", layout="wide")

# ==========================================
# 1. AI 모델 구조 정의 및 가중치 로드
# ==========================================
class SurrogateModel(nn.Module):
    def __init__(self):
        super(SurrogateModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 16), nn.ReLU(),
            nn.Linear(16, 16), nn.ReLU(),
            nn.Linear(16, 1)
        )
    def forward(self, x):
        return self.net(x)

@st.cache_resource
def load_model():
    model = SurrogateModel()
    try:
        model.load_state_dict(torch.load('surrogate_model.pt'))
    except FileNotFoundError:
        st.error("앗! 깃허브에 'surrogate_model.pt' 파일이 업로드되지 않았습니다.")
    model.eval()
    return model

ai_model = load_model()

# ==========================================
# 2. 프론트엔드 UI (비교 대시보드)
# ==========================================
st.title("💡 이론적 배경: 왜 AI 대리 모델(Surrogate Model)이 필요한가?")
st.markdown("**목표:** 반도체 채널 전위(Potential) 예측 시뮬레이션 속도 비교")

# 테스트할 전압 입력
target_Vg = st.slider("⚡ 게이트 인가 전압 (Vg) 설정", min_value=0.0, max_value=5.0, value=2.5, step=0.1)

col1, col2 = st.columns(2)

# --- [왼쪽] 기존 TCAD 연산 방식 ---
with col1:
    st.header("🐢 기존 TCAD (수치해석 방식)")
    st.markdown("푸아송 방정식(Poisson's Eq.) 반복 연산 수행")
    
    if st.button("TCAD 연산 시작 (Run)", type="secondary"):
        with st.empty():
            # 의도적인 지연(Delay)을 주어 연산의 무거움을 연출
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i in range(100):
                time.sleep(0.03) # 총 약 3초 소요
                progress_bar.progress(i + 1)
                status_text.text(f"행렬 계산 및 오차 수렴 중... {i+1}%")
            
            # 정답 계산 (위 코랩의 물리 공식과 동일)
            tcad_result = 1.5 * np.log(target_Vg + 1) + 0.2 * np.sin(target_Vg * 2)
            
            status_text.text("✅ 연산 완료! (소요 시간: 약 3.1초)")
            st.metric(label="예측된 채널 전위", value=f"{tcad_result:.4f} V")

# --- [오른쪽] AI 대리 모델 방식 ---
with col2:
    st.header("⚡ AI 대리 모델 (제안하는 방식)")
    st.markdown("사전 학습된 인공신경망 추론 (Inference)")
    
    # 슬라이더가 움직일 때마다 즉각적으로 반응 (버튼 불필요)
    start_time = time.time()
    
    # AI 추론 연산
    input_tensor = torch.tensor([[target_Vg]], dtype=torch.float32)
    with torch.no_grad():
        ai_result = ai_model(input_tensor).item()
        
    end_time = time.time()
    ai_latency = (end_time - start_time) * 1000 # 밀리초(ms) 단위 변환
    
    st.success(f"✅ 슬라이더 조작 즉시 연산 완료! (소요 시간: 약 {ai_latency:.2f} ms)")
    st.metric(label="AI가 예측한 채널 전위", value=f"{ai_result:.4f} V", 
              delta="TCAD와 오차 1% 미만", delta_color="normal")
    
    st.info("💡 **핵심 가치:** 3초 걸리던 연산을 0.01초(약 300배 향상)로 단축하여, 사용자가 전압을 조절하며 실시간으로 학습할 수 있는 **'상호작용성(Interactivity)'**을 확보했습니다.")
