import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
from openai import OpenAI

# --- [1. 페이지 설정 및 초기화] ---
st.set_page_config(page_title="고도화된 멀티 AI 투자 에이전트", layout="wide")
st.title("🚀 Advanced Multi-Agent Investment System")
st.caption("실시간 기술적 지표 연산, 파이낸셜 데이터 추출 및 OpenAI 멀티 에이전트 오케스트레이션")

# --- [2. 사이드바: 설정 및 API 키 입력] ---
st.sidebar.header("🔑 인증 및 에이전트 설정")
openai_api_key = st.sidebar.text_input("OpenAI API Key 입력", type="password", placeholder="sk-...")

st.sidebar.markdown("---")
st.sidebar.header("🔍 분석 대상 설정")
ticker_input = st.sidebar.text_input("종목 코드 입력 (국내는 .KS / .KQ 붙임)", value="000660.KS")
start_date = st.sidebar.date_input("조회 시작일", datetime.date(2025, 1, 1))
end_date = st.sidebar.date_input("조회 종료일", datetime.date.today())

st.sidebar.markdown("---")
st.sidebar.subheader("🤖 에이전트 파라미터")
llm_model = st.sidebar.selectbox("사용할 LLM 모델", ["gpt-4o", "gpt-4o-mini"])
temperature_val = st.sidebar.slider("에이전트 창의성 (Temperature)", 0.0, 1.0, 0.2, 0.1)

btn_run = st.sidebar.button("멀티 에이전트 가동", use_container_width=True, type="primary")

# --- [3. 핵심 연산 함수 (기술적 지표 계산)] ---
def calculate_technical_indicators(df):
    """Pandas를 이용해 RSI와 MACD 지표를 직접 계산합니다."""
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA60'] = df['Close'].rolling(window=60).mean()
    
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-9) 
    df['RSI'] = 100 - (100 / (1 + rs))
    
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['Signal']
    
    return df

# --- [4. OpenAI 멀티 에이전트 프롬프트 체인 함수] ---
def run_llm_agent(role_system_prompt, user_content):
    """OpenAI API를 사용하여 개별 에이전트의 분석을 수행합니다."""
    if not openai_api_key:
        return "⚠️ 사이드바에 OpenAI API Key를 입력하셔야 실제 AI 분석이 진행됩니다."
    
    try:
        client = OpenAI(api_key=openai_api_key)
        response = client.chat.completions.create(
            model=llm_model,
            messages=[
                {"role": "system", "content": role_system_prompt + "\n모든 답변은 전문적이고 객관적인 한국어로 작성하세요."},
                {"role": "user", "content": user_content}
            ],
            temperature=temperature_val
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ 에이전트 가동 중 오류 발생: {str(e)}"

# --- [5. 메인 실행 로직] ---
if btn_run or ticker_input:
    with st.spinner("🎯 실시간 마켓 데이터를 수집하고 멀티 에이전트를 조율 중입니다..."):
        try:
            # 데이터 수집
            stock = yf.Ticker(ticker_input)
            hist = stock.history(start=start_date, end=end_date)
            info = stock.info
            news_list = stock.news
            
            if hist.empty:
                st.error("❌ 주가 데이터를 가져오지 못했습니다. 종목 코드가 올바른지 확인하세요.")
                st.stop()
                
            # 기술적 지표 연산 적용
            hist = calculate_technical_indicators(hist)
            latest_data = hist.iloc[-1]
            
            # 재무제표 가공 (대차대조표 추출)
            balance_sheet = stock.balance_sheet
            financials_summary = ""
            if not balance_sheet.empty:
                avail_rows = balance_sheet.index[:12]
                financials_summary = balance_sheet.loc[avail_rows, balance_sheet.columns[:2]].to_string()
            
            # 뉴스 데이터 가공
            news_summary_text = ""
            if news_list:
                for idx, n in enumerate(news_list[:5]):
                    news_summary_text += f"[{idx+1}] 제목: {n.get('title')}\n출처: {n.get('publisher')}\n\n"
            else:
                news_summary_text = "최근 관련 뉴스가 존재하지 않습니다."

            # 기업 기본 헤더 출력
            company_name = info.get('longName', info.get('shortName', ticker_input))
            current_price = info.get('currentPrice', latest_data['Close'])
            currency = info.get('currency', 'KRW')
            
            # 메인 대시보드 상단 요약 지표 (Metrics)
            st.subheader(f"📊 {company_name} ({ticker_input}) 실시간 데이터 기반 종합 대시보드")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("현재가 / 최종종가", f"{current_price:,.0f} {currency}")
            col2.metric("20일 이동평균선", f"{latest_data['MA20']:,.0f} {currency}")
            col3.metric("RSI (14)", f"{latest_data['RSI']:.2f}")
            col4.metric("MACD Score", f"{latest_data['MACD']:.2f}")
            
            # 탭 구조 정의
            tab1, tab2, tab3, tab4 = st.tabs([
                "📈 고도화된 기술적 차트 분석", 
                "📑 대차대조표 및 재무 지표", 
                "📰 실시간 뉴스 피드 데이터",
                "🧠 멀티 AI 에이전트 종합 브리핑"
            ])
            
            # ----------------------------------------------------
            # Tab 1: 기술적 차트 분석 (Plotly Subplots 수정 완료)
            # ----------------------------------------------------
            with tab1:
                st.subheader("📊 멀티 인디케이터 기술적 분석 차트")
                
                # 3단 서브플롯 차트 구성
                fig = make_subplots(
                    rows=3, cols=1, 
                    shared_xaxes=True, 
                    vertical_spacing=0.05, 
                    subplot_titles=('Price & Moving Averages', 'MACD (12, 26, 9)', 'RSI (14)'),
                    row_heights=[0.5, 0.25, 0.25]
                )
                
                # 행 1: 캔들스틱 및 이동평균선
                fig.add_trace(go.Candlestick(
                    x=hist.index, open=hist['Open'], high=hist['High'], low=hist['Low'], close=hist['Close'], name='주가'
                ), row=1, col=1)
                fig.add_trace(go.Scatter(x=hist.index, y=hist['MA20'], mode='lines', name='20일 이평선', line=dict(color='orange', width=1.5)), row=1, col=1)
                fig.add_trace(go.Scatter(x=hist.index, y=hist['MA60'], mode='lines', name='60일 이평선', line=dict(color='blue', width=1.5)), row=1, col=1)
                
                # 행 2: MACD & Signal
                fig.add_trace(go.Scatter(x=hist.index, y=hist['MACD'], mode='lines', name='MACD', line=dict(color='red')), row=2, col=1)
                fig.add_trace(go.Scatter(x=hist.index, y=hist['Signal'], mode='lines', name='Signal', line=dict(color='green', dash='dot')), row=2, col=1)
                fig.add_trace(go.Bar(x=hist.index, y=hist['MACD_Hist'], name='Hist'), row=2, col=1)
                
                # 행 3: RSI
                fig.add_trace(go.Scatter(x=hist.index, y=hist['RSI'], mode='lines', name='RSI', line=dict(color='purple')), row=3, col=1)
                fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="blue", row=3, col=1)
                
                fig.update_layout(xaxis_rangeslider_visible=False, height=700, margin=dict(t=50, b=20, l=10, r=10))
                st.plotly_chart(fig, use_container_width=True)
                
            # ----------------------------------------------------
            # Tab 2: 대차대조표 분석
            # ----------------------------------------------------
            with tab2:
                st.subheader("📑 대차대조표 (Balance Sheet) 원 데이터")
                if not balance_sheet.empty:
                    st.dataframe(balance_sheet, use_container_width=True)
                else:
                    st.warning("이 종목의 대차대조표 데이터를 호출할 수 없습니다.")
                    
            # ----------------------------------------------------
            # Tab 3: 뉴스 피드 데이터
            # ----------------------------------------------------
            with tab3:
                st.subheader("📰 마켓 실시간 관련 뉴스 리스트")
                if news_list:
                    for n in news_list[:8]:
                        with st.expander(n.get('title', '제목 없음')):
                            st.write(f"**출처:** {n.get('publisher')} | **발행시간:** {datetime.datetime.fromtimestamp(n.get('providerPublishTime', 0))}")
                            st.markdown(f"[뉴스 원문 링크 이동]({n.get('link')})")
                else:
                    st.info("수집된 실시간 뉴스가 없습니다.")

            # ----------------------------------------------------
            # Tab 4: 멀티 AI 에이전트 종합 브리핑
            # ----------------------------------------------------
            with tab4:
                st.subheader("🧠 데이터 기반 멀티 에이전트 협업 분석 체인")
                
                if not openai_api_key:
                    st.warning("⚠️ AI 에이전트의 정밀 브리핑을 보려면 사이드바에 OpenAI API Key를 입력하고 다시 가동해 주세요.")
                else:
                    tech_context = f"현재가: {current_price}, 20일이평선: {latest_data['MA20']:.2f}, 60일이평선: {latest_data['MA60']:.2f}, RSI: {latest_data['RSI']:.2f}, MACD: {latest_data['MACD']:.2f}, MACD 시그널: {latest_data['Signal']:.2f}"
                    
                    with st.status("1. 차트/기술적 에이전트 분석 중...", expanded=False) as status:
                        p_tech = "당신은 금융 시장의 리스크 관리 및 계량 분석 전문가인 '차트 기술 분석 에이전트'입니다. 제공된 주가 변동성 지표(MA, RSI, MACD 등)의 수치를 수학적, 통계학적 관점에서 철저히 분석하여 현재 주가가 과열 상태인지, 지지선 근처인지 명확한 결론을 도출하세요."
                        res_tech = run_llm_agent(p_tech, f"종목명: {company_name}\n지표 데이터: {tech_context}")
                        st.markdown(res_tech)
                        status.update(label="기술적 에이전트 분석 완료", state="complete")
                        
                    with st.status("2. 재무/대차대조표 에이전트 분석 중...", expanded=False) as status:
                        p_fund = "당신은 기업의 내재 가치를 평가하는 '재무 분석 에이전트'입니다. 제공되는 대차대조표의 자산, 부채, 자본 구조를 바탕으로 기업의 단기 유동성 리스크, 장기 재무 건전성 및 공정 가치 대비 안정성을 정량적으로 비판 평가하세요."
                        res_fund = run_llm_agent(p_fund, f"종목명: {company_name}\n대차대조표 상위 데이터:\n{financials_summary}")
                        st.markdown(res_fund)
                        status.update(label="재무 분석 에이전트 분석 완료", state="complete")
                        
                    with st.status("3. 뉴스/센티먼트 에이전트 분석 중...", expanded=False) as status:
                        p_news = "당신은 시장의 심리와 비정형 데이터를 정제하는 '뉴스 센티먼트 분석 에이전트'입니다. 최근 5개 주요 뉴스의 헤드라인을 바탕으로, 해당 종목을 둘러싼 거시경제 흐름, 공급망 리스크 등 정성적 요소를 분류하고 긍정/부정 스코어를 논리적으로 제시하세요."
                        res_news = run_llm_agent(p_news, f"종목명: {company_name}\n최신 뉴스 목록:\n{news_summary_text}")
                        st.markdown(res_news)
                        status.update(label="뉴스 분석 에이전트 분석 완료", state="complete")
                        
                    st.markdown("---")
                    st.markdown("### 🏛️ 수석 투자 최고책임자(CIO) 에이전트의 종합 변론")
                    
                    p_cio = """당신은 독립 자산운용사의 수석 투자 최고책임자(CIO)입니다. 
                    하위 3명의 전문 에이전트(기술 차트, 대차대조표 재무, 뉴스 센티먼트)가 작성한 개별 보고서를 냉철하게 결합하고, 
                    여기에 글로벌 거시경제 흐름을 유기적으로 엮어 최종 투자 자문 브리핑을 작성해야 합니다. 
                    향후 1~3개월간 취해야 할 구체적인 포지션 전략(분할 매수, 관망, 비중 축소 등)과 핵심 모니터링 지표를 명확히 제시하십시오."""
                    
                    combined_input = f"""
                    [대상 기업] {company_name} ({ticker_input})
                    [1. 기술적 분석 결과]\n{res_tech}\n\n
                    [2. 재무제표 분석 결과]\n{res_fund}\n\n
                    [3. 뉴스 및 센티먼트 분석 결과]\n{res_news}
                    """
                    
                    res_cio = run_llm_agent(p_cio, combined_input)
                    st.write(res_cio)
                    
        except Exception as e:
            st.error(f"⚠️ 대시보드 구동 중 에러가 발생했습니다: {str(e)}")
            st.info("팁: 일시적인 데이터 제공사 오류일 수 있으니 잠시 후 다시 시도해 보세요.")
