import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as px
from plotly.subplots import make_subplots
import datetime

# --- [이름 정의 및 페이지 설정] ---
st.set_page_config(page_title="나만의 AI 투자 에이전트", layout="wide")
st.title("🤖 My AI Investment Agent Dashboard")
st.caption("뉴스 분석, 기술적 지표, 대차대조표 및 거시경제를 종합 분석하는 AI 비서")

# --- [사이드바: 사용자 입력 컨트롤] ---
st.sidebar.header("🔍 분석 설정")
ticker_input = st.sidebar.text_input("종목 코드 입력 (예: 삼성전자는 005930.KS / 애플은 AAPL)", value="005930.KS")
start_date = st.sidebar.date_input("조회 시작일", datetime.date(2025, 1, 1))
end_date = st.sidebar.date_input("조회 종료일", datetime.date.today())

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ AI 에이전트 가동")
run_macro_agent = st.sidebar.checkbox("거시경제 에이전트 활성화", value=True)
run_news_agent = st.sidebar.checkbox("뉴스/리포트 감성분석 활성화", value=True)

# 데이터 로드 버튼
btn_run = st.sidebar.button("종목 분석 시작", use_container_width=True)

# --- [가상의 AI 에이전트 기능 모듈 (API 연동용 가상 함수)] ---
def ai_macro_analysis():
    # 실제 구현 시: FRED API 등으로 국고채 금리, 환율 데이터를 받아 LLM 프롬프트에 전달
    return """
    **[거시경제 에이전트 의견]**
    현재 미국 연준의 금리 동결 기조와 국내 수출 지표 반등이 맞물려 전반적인 IT/제조업 섹터에 긍정적인 환경이 조성되고 있습니다. 
    다만 원/달러 환율 변동성이 존재하므로 외국인 수급 추이를 주시할 필요가 있습니다.
    """

def ai_news_and_analyst_summary(ticker):
    # 실제 구현 시: 네이버 뉴스 API 또는 증권사 크롤링 데이터를 LLM에 요약 요청
    return f"""
    **[뉴스 & 애널리스트 종합 의견 ({ticker})]**
    - **주요 뉴스 요약:** 차세대 제품 양산 수율 안정화 궤도 진입 뉴스 및 글로벌 빅테크 기업 공급망 다변화 수혜 기대감이 지배적입니다.
    - **애널리스트 컨센서스:** 목표주가 상향 조정 추세. 2분기 실적 저점 통과 후 하반기 턴어라운드 강도가 강할 것이라는 의견이 다수입니다.
    - **종합 감성 점수:** **긍정 (82%)**
    """

# --- [메인 대시보드 로직] ---
if btn_run or ticker_input:
    with st.spinner("금융 데이터를 수집하고 AI 에이전트를 구동 중입니다..."):
        try:
            # 1. 주식 데이터 및 기업 정보 가져오기
            stock = yf.Ticker(ticker_input)
            hist = stock.history(start=start_date, end=end_date)
            info = stock.info
            
            # 기업 기본 정보 표시
            company_name = info.get('longName', ticker_input)
            st.subheader(f"📊 {company_name} ({ticker_input}) 분석 리포트")
            
            # --- [탭 구조 분할] ---
            tab1, tab2, tab3, tab4 = st.tabs([
                "🌐 거시경제 흐름 & 뉴스 요약", 
                "📈 차트 기술적 분석", 
                "📑 대차대조표 & 재무", 
                "🧠 AI 종합 투자 의견"
            ])
            
            # ----------------------------------------------------
            # Tab 1: 거시경제 흐름 & 뉴스 요약
            # ----------------------------------------------------
            with tab1:
                st.subheader("🌐 글로벌 거시경제 흐름")
                if run_macro_agent:
                    st.info(ai_macro_analysis())
                else:
                    st.warning("거시경제 에이전트가 비활성화되어 있습니다.")
                
                st.markdown("---")
                st.subheader("📰 개별 종목 뉴스 및 애널리스트 리포트 요약")
                if run_news_agent:
                    st.success(ai_news_and_analyst_summary(ticker_input))
                else:
                    st.warning("뉴스 분석 에이전트가 비활성화되어 있습니다.")

            # ----------------------------------------------------
            # Tab 2: 차트 기술적 분석
            # ----------------------------------------------------
            with tab2:
                st.subheader("📈 기술적 지표 시각화 (OHLCV & 이동평균선)")
                
                if not hist.empty:
                    # 간단한 이동평균선(MA) 계산
                    hist['MA20'] = hist['Close'].rolling(window=20).mean()
                    hist['MA60'] = hist['Close'].rolling(window=60).mean()
                    
                    # Plotly를 이용한 캔들스틱 및 거래량 차트 생성
                    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                                       vertical_spacing=0.1, subplot_titles=('주가 추이', '거래량'),
                                       row_width=[0.3, 0.7])
                    
                    # 캔들스틱 추가
                    fig.add_trace(px.Candlestick(x=hist.index, open=hist['Open'], high=hist['High'],
                                                 low=hist['Low'], close=hist['Close'], name='주가').data[0], row=1, col=1)
                    
                    # 이동평균선 추가
                    fig.add_trace(px.line(x=hist.index, y=hist['MA20'], name='20일 이평선').data[0], row=1, col=1)
                    fig.add_trace(px.line(x=hist.index, y=hist['MA60'], name='60일 이평선').data[0], row=1, col=1)
                    
                    # 거래량 추가
                    fig.add_trace(px.bar(x=hist.index, y=hist['Volume'], name='거래량').data[0], row=2, col=1)
                    
                    fig.update_layout(xaxis_rangeslider_visible=False, height=500, margin=dict(t=30, b=10))
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # 간단한 기술적 수치 서머리
                    latest_close = hist['Close'].iloc[-1]
                    latest_ma20 = hist['MA20'].iloc[-1]
                    st.write(f"**최종 종가:** {latest_close:,.0f}원 / **20일 이동평균선:** {latest_ma20:,.0f}원")
                else:
                    st.error("차트 데이터를 불러오지 못했습니다. 종목 코드를 확인해 주세요.")

            # ----------------------------------------------------
            # Tab 3: 대차대조표 & 재무 분석
            # ----------------------------------------------------
            with tab3:
                st.subheader("📑 대차대조표 (Balance Sheet) 및 주요 재무제표")
                
                # yfinance에서 대차대조표 가져오기
                balance_sheet = stock.balance_sheet
                
                if not balance_sheet.empty:
                    # 최근 3개년 데이터 보여주기
                    st.dataframe(balance_sheet.iloc[:15, :3], use_container_width=True)
                    
                    st.markdown("💡 *위 테이블은 yfinance에서 원 데이터 추출 후 가공한 지표입니다. 실제 구동 시 OpenDART 등과 매핑하여 한글 항목명으로 치환하면 가독성이 더욱 높아집니다.*")
                else:
                    st.warning("이 종목의 대차대조표 데이터를 제공하지 않거나 불러오지 못했습니다.")

            # ----------------------------------------------------
            # Tab 4: AI 종합 투자 의견
            # ----------------------------------------------------
            with tab4:
                st.subheader("🧠 수석 AI 에이전트의 종합 최종 브리핑")
                
                # 정량적 데이터(차트, 재무)와 정성적 데이터(뉴스, 매크로)를 취합하는 프롬프트 결과 자리
                st.write("""
                위에서 수집된 **기술적 분석 수치, 대차대조표 건전성, 뉴스 감성 분석, 매크로 지표**를 
                종합적으로 판단한 결과, 본 종목은 단기적인 매크로 변동성 대비 강한 가치 방어력을 보여주고 있습니다. 
                
                **[에이전트 권장 액션 가이드]**
                1. **진입 시점:** 주가가 20일 이동평균선 부근까지 건전한 조정을 줄 때 분할 매수 관점 유효.
                2. **리스크 요인:** 대차대조표상 단기부채 비율 변동 여부 및 환율 1,350원 돌파 여부 모니터링 필요.
                """)
                
        except Exception as e:
            st.error(f"데이터를 가져오는 중 오류가 발생했습니다: {e}")
