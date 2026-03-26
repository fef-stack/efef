import streamlit as st
import numpy as np
import plotly.graph_objects as go

# 페이지 기본 설정
st.set_page_config(page_title="Vibe Coding: DRAM Simulator", layout="wide")

# ==========================================
# 0. 세션 상태 초기화 (퀴즈 진행 상황 저장용)
# ==========================================
if 'quiz_passed' not in st.session_state:
    st.session_state.quiz_passed = False

# ==========================================
# 1. 도입부 & 퀴즈 섹션 (호기심 유발)
# ==========================================
st.title("📱 AI 바이브 코딩: 반도체 메모리 시뮬레이터")
st.markdown("---")

# 퀴즈를 통과하지 않았을 때만 퀴즈 화면 표시
if not st.session_state.quiz_passed:
    st.header("🤔 Step 1. 실전 면접 퀴즈")
    st.info("**Q. 고사양 게임을 돌려 스마트폰이 뜨거워지면, DRAM 메모리 내부에서는 어떤 물리적 변화가 가장 치명적으로 발생할까요?**")
    
    answer = st.radio(
        "정답을 선택하세요:",
        ["선택하세요",
         "1. 전자의 이동 속도가 빨라져 연산이 더 잘 된다.",
         "2. 열에너지(kT)를 얻은 전자가 방벽을 넘어 탈출하는 '누설 전류'가 기하급수적으로 증가해 데이터가 소실된다.",
         "3. 메모리 용량이 일시적으로 늘어난다."]
    )
    
    if st.button("정답 확인 및 시뮬레이션 가동 🚀"):
        if answer.startswith("2"):
            st.success("🎉 정답입니다! 열에너지에 의한 누설 전류가 핵심입니다. 이제 시뮬레이션으로 직접 확인해 볼까요?")
            st.session_state.quiz_passed = True
            st.rerun() # 화면 새로고침하여 시뮬레이션 띄우기
        elif answer == "선택하세요":
            st.warning("정답을 선택해 주세요.")
        else:
            st.error("오답입니다. 스마트폰이 뜨거워지면 배터리가 빨리 닳고 버벅이는 이유를 생각해 보세요!")

# ==========================================
# 2. 메인 시뮬레이션 섹션 (퀴즈 정답 맞춘 후 표시)
# ==========================================
if st.session_state.quiz_passed:
    st.header("🔬 Step 2. DRAM 누설 전류 물리 시뮬레이션")
    st.markdown("**아레니우스 방정식(Arrhenius Equation)**을 기반으로 온도 변화에 따른 전압 강하를 실시간으로 계산합니다.")
    
    col_control, col_graph = st.columns([1, 2.5])
    
    with col_control:
        st.subheader("🎛️ 환경 설정")
        # 온도 조절 슬라이더
        temperature_c = st.slider("칩 온도 (Temperature) [°C]", min_value=20, max_value=105, value=25, step=5)
        temperature_k = temperature_c + 273.15 # 켈빈 온도로 변환
        
        # 다시 풀기 버튼
        st.markdown("<br><br>", unsafe_allow_html=True)
        if st.button("🔄 퀴즈 다시 풀기"):
            st.session_state.quiz_passed = False
            st.rerun()

    # 물리 상수 및 DRAM 파라미터 세팅
    k_B = 8.617e-5  # 볼츠만 상수 (eV/K)
    E_a = 0.6       # 활성화 에너지 (eV)
    V_initial = 1.2 # 초기 충전 전압 (V) - 논리 '1'
    V_ref = 0.6     # 기준 전압 (V) - 이 밑으로 떨어지면 데이터 '0'으로 인식
    C_cell = 10e-15 # 셀 커패시턴스 (F)
    
    # 누설 전류 계산 (Arrhenius Model)
    # I_leak = I_0 * exp(-E_a / kT)
    I_0 = 1e-6 # Reference current 
    I_leak = I_0 * np.exp(-E_a / (k_B * temperature_k))
    
    # 시간에 따른 전압 변화 계산 (V = V0 - (I*t)/C)
    time_ms = np.linspace(0, 100, 200) # 0 ~ 100ms
    voltage_drop = V_initial - (I_leak * (time_ms * 1e-3)) / C_cell
    voltage_drop = np.maximum(voltage_drop, 0) # 전압은 0 이하로 떨어지지 않음
    
    # 데이터 소실 시점(Retention Time) 계산
    retention_time_idx = np.where(voltage_drop < V_ref)[0]
    if len(retention_time_idx) > 0:
        retention_ms = time_ms[retention_time_idx[0]]
        status_text = f"🚨 데이터 소실 발생! ({retention_ms:.1f} ms)"
        status_color = "red"
    else:
        retention_ms = "> 100"
        status_text = "✅ 데이터 안전 (Refresh 주기 내)"
        status_color = "#00FF00"

    with col_graph:
        # Plotly를 이용한 동적 그래프 렌더링
        fig = go.Figure()
        
        # 실시간 전압 변화 곡선
        fig.add_trace(go.Scatter(x=time_ms, y=voltage_drop, mode='lines', 
                                 line=dict(color='cyan', width=4), name='셀 전압 (V_cell)'))
        
        # 데이터 판독 기준선 (V_ref)
        fig.add_trace(go.Scatter(x=[0, 100], y=[V_ref, V_ref], mode='lines', 
                                 line=dict(color='red', width=2, dash='dash'), name='판독 임계 전압 (0.6V)'))
        
        # 위험 구역 표시
        fig.add_hrect(y0=0, y1=V_ref, fillcolor="red", opacity=0.1, layer="below", line_width=0)
        
        fig.update_layout(
            title=dict(text=f"현재 칩 온도: {temperature_c}°C ➔ {status_text}", font=dict(color=status_color, size=20)),
            xaxis_title="시간 (milliseconds)",
            yaxis_title="DRAM 셀 전압 (Voltage)",
            yaxis=dict(range=[0, 1.3]),
            template="plotly_dark",
            margin=dict(l=20, r=20, t=50, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # 핵심 지표 표시
        m1, m2 = st.columns(2)
        m1.metric("예상 누설 전류 (I_leak)", f"{I_leak * 1e15:.2f} fA")
        m2.metric("데이터 보존 시간 (Retention Time)", f"{retention_ms} ms" if isinstance(retention_ms, float) else retention_ms)
