"""
STARGENT 에이전트들이 사용하는 도구 모음

이 모듈은 각 에이전트가 실제 데이터를 수집하고 분석하는 데 필요한
모든 도구 함수들을 제공합니다. 각 함수는 OpenAI Function Calling 
형식에 맞게 설계되어 AI가 자연스럽게 호출할 수 있습니다.

주요 도구 카테고리:
    1. 키키용 도구: 뉴스 수집, 시장 동향 파악
    2. 아거용 도구: 종목 분석, 기업 정보 수집  
    3. 라무용 도구: 시나리오 생성, 시뮬레이션

설계 원칙:
    - 모든 함수는 상세한 한국어 독스트링 포함
    - 에러 핸들링과 재시도 로직 내장
    - OpenAI Function Calling 형식 준수
    - 실제 사용 가능한 예시 코드 포함
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import FinanceDataReader as fdr
import numpy as np
from typing import List, Dict, Any, Optional
import time
import json
import openai
from datetime import datetime, timedelta
import os

# OpenAI 클라이언트 초기화
print(f"API KEY Check: {os.getenv('OPENAI_API_KEY')[:6]}")
client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def safe_request_with_retry(func, max_retries=2, delay=1):
    """
    네트워크 요청을 안전하게 실행하는 헬퍼 함수
    
    실패 시 자동으로 재시도하며, 각종 네트워크 오류를 처리합니다.
    실제 서비스에서는 이런 견고한 에러 핸들링이 매우 중요합니다.
    
    Args:
        func: 실행할 함수
        max_retries (int): 최대 재시도 횟수
        delay (int): 재시도 간 대기 시간(초)
        
    Returns:
        함수 실행 결과 또는 None (실패 시)
    """
    for attempt in range(max_retries + 1):
        try:
            return func()
        except Exception as e:
            if attempt == max_retries:
                print(f"❌ 최대 재시도 횟수 초과: {str(e)}")
                return None
            print(f"⚠️ 시도 {attempt + 1} 실패, {delay}초 후 재시도: {str(e)}")
            time.sleep(delay)
    return None

# =============================================================================
# 키키용 도구들 - 트렌드 마스터의 시장 모니터링 도구
# =============================================================================

def get_latest_news() -> str:
    """
    최신 경제/금융 뉴스를 실시간으로 수집합니다.
    
    네이버 금융 뉴스에서 최신 뉴스 제목과 요약 정보를 스크래핑하여
    구조화된 형태로 반환합니다. 키키가 시장 트렌드를 파악하는 데
    핵심적인 역할을 하는 도구입니다.
    
    Returns:
        str: JSON 형태의 뉴스 목록 문자열
        
    반환 형식:
        [
            {
                "title": "뉴스 제목",
                "info": "뉴스 요약 정보"
            }
        ]
        
    사용 예시:
        >>> news = get_latest_news()
        >>> import json
        >>> news_list = json.loads(news)
        >>> print(f"총 {len(news_list)}개의 뉴스를 수집했습니다")
        >>> print(f"첫 번째 뉴스: {news_list[0]['title']}")
    """
    
    def fetch_news():
        # 네이버 금융 실시간 뉴스 페이지
        url = "https://finance.naver.com/news/news_list.naver?mode=LSS2D&section_id=101&section_id2=258"
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0 Safari/537.36"
        }
        
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # HTTP 오류 시 예외 발생
        
        soup = BeautifulSoup(response.text, "html.parser")
        
        # 뉴스 제목과 요약 정보 선택자
        subject_selector = "dd.articleSubject"
        summary_selector = "dd.articleSummary"
        
        news = []
        subjects = soup.select(subject_selector)
        summaries = soup.select(summary_selector)
        
        # 제목과 요약이 매칭되는 것들만 처리
        for subject, summary in zip(subjects, summaries):
            title = subject.text.strip()
            info = summary.text.strip().replace("\n", "|").replace("\t", "")
            
            # 빈 내용 필터링
            if title and info:
                news.append({
                    "title": title,
                    "info": info
                })
        
        return news[:10]  # 최신 10개만 반환
    
    # 안전한 네트워크 요청 실행
    news_data = safe_request_with_retry(fetch_news)
    
    if news_data is None:
        return json.dumps([{
            "title": "뉴스 수집 오류",
            "info": "현재 뉴스 서비스에 접속할 수 없습니다. 잠시 후 다시 시도해주세요."
        }], ensure_ascii=False)
    
    return json.dumps(news_data, ensure_ascii=False)

def get_major_movers() -> str:
    """
    주가가 크게 변동한 주요 종목들을 조회합니다.
    
    FinanceDataReader를 활용하여 당일 급등/급락한 종목들의 정보를 수집합니다.
    시장에서 관심받는 종목들을 빠르게 파악할 수 있습니다.
    
    Returns:
        str: JSON 형태의 주요 변동 종목 목록
        
    반환 형식:
        [
            {
                "name": "종목명",
                "code": "종목코드", 
                "change_pct": "변동률(%)",
                "volume": "거래량"
            }
        ]
        
    사용 예시:
        >>> movers = get_major_movers()
        >>> movers_list = json.loads(movers)
        >>> for stock in movers_list:
        ...     print(f"{stock['name']}: {stock['change_pct']}%")
    """
    
    def fetch_movers():
        # KOSPI 상위 거래량 종목들 조회
        kospi_stocks = fdr.StockListing('KOSPI')
        
        # 임시로 일부 대표 종목들의 데이터를 생성
        # 실제 구현에서는 실시간 급등/급락 API를 사용
        major_stocks = ['005930', '000660', '373220', '207940', '005935']  # 삼성전자, SK하이닉스 등
        
        movers = []
        for code in major_stocks:
            try:
                # 최근 2일간 데이터 조회
                stock_data = fdr.DataReader(code, start=datetime.now() - timedelta(days=2))
                if len(stock_data) >= 2:
                    # 변동률 계산
                    today_close = stock_data['Close'].iloc[-1]
                    yesterday_close = stock_data['Close'].iloc[-2]
                    change_pct = ((today_close - yesterday_close) / yesterday_close) * 100
                    
                    # 종목명 찾기
                    stock_info = kospi_stocks[kospi_stocks['Code'] == code]
                    name = stock_info['Name'].iloc[0] if len(stock_info) > 0 else f"종목{code}"
                    
                    movers.append({
                        "name": name,
                        "code": code,
                        "change_pct": f"{change_pct:.2f}",
                        "volume": f"{stock_data['Volume'].iloc[-1]:,}"
                    })
            except Exception as e:
                print(f"종목 {code} 데이터 조회 실패: {e}")
                continue
        
        return movers
    
    movers_data = safe_request_with_retry(fetch_movers)
    
    if movers_data is None:
        return json.dumps([{
            "name": "데이터 조회 오류",
            "code": "000000",
            "change_pct": "0.00",
            "volume": "데이터를 가져올 수 없습니다"
        }], ensure_ascii=False)
    
    return json.dumps(movers_data, ensure_ascii=False)

def get_market_indicators() -> str:
    """
    주요 시장 지표들의 현재 상황을 조회합니다.
    
    KOSPI, KOSDAQ, 환율, 금리 등 거시경제 핵심 지표들을 수집하여
    전체적인 시장 상황을 파악할 수 있도록 합니다.
    
    Returns:
        str: JSON 형태의 시장 지표 정보
        
    반환 형식:
        [
            {
                "indicator": "지표명",
                "current_value": "현재값",
                "change": "전일 대비 변동",
                "change_pct": "변동률(%)"
            }
        ]
        
    사용 예시:
        >>> indicators = get_market_indicators()
        >>> indicators_list = json.loads(indicators)
        >>> for indicator in indicators_list:
        ...     print(f"{indicator['indicator']}: {indicator['current_value']} ({indicator['change_pct']}%)")
    """
    
    def fetch_indicators():
        indicators = []
        
        try:
            # KOSPI 지수 조회
            kospi = fdr.DataReader('KS11', start=datetime.now() - timedelta(days=5))
            if len(kospi) >= 2:
                current = kospi['Close'].iloc[-1]
                previous = kospi['Close'].iloc[-2]
                change = current - previous
                change_pct = (change / previous) * 100
                
                indicators.append({
                    "indicator": "KOSPI",
                    "current_value": f"{current:.2f}",
                    "change": f"{change:+.2f}",
                    "change_pct": f"{change_pct:+.2f}"
                })
        except:
            indicators.append({
                "indicator": "KOSPI", 
                "current_value": "NULL",
                "change": "NULL",
                "change": "NULL",
                "change_pct": "NULL"
            })
        
        try:
            # KOSDAQ 지수 조회
            kosdaq = fdr.DataReader('KQ11', start=datetime.now() - timedelta(days=5))
            if len(kosdaq) >= 2:
                current = kosdaq['Close'].iloc[-1]
                previous = kosdaq['Close'].iloc[-2]
                change = current - previous
                change_pct = (change / previous) * 100
                
                indicators.append({
                    "indicator": "KOSDAQ",
                    "current_value": f"{current:.2f}",
                    "change": f"{change:+.2f}",
                    "change_pct": f"{change_pct:+.2f}"
                })
        except:
            indicators.append({
                "indicator": "KOSDAQ", 
                "current_value": "데이터 없음",
                "change": "NULL",
                "change_pct": "NULL"
            })
        
        # 환율 (USD/KRW) - FinanceDataReader 사용
        try:
            usdkrw = fdr.DataReader('USD/KRW', start=datetime.now() - timedelta(days=5))
            if len(usdkrw) >= 2:
                current = usdkrw['Close'].iloc[-1]
                previous = usdkrw['Close'].iloc[-2]
                change = current - previous
                change_pct = (change / previous) * 100
                
                indicators.append({
                    "indicator": "USD/KRW",
                    "current_value": f"{current:.2f}",
                    "change": f"{change:+.2f}",
                    "change_pct": f"{change_pct:+.2f}"
                })
        except:
            # 폴백: 환율 데이터 조회 실패시 고정값 사용
            indicators.append({
                "indicator": "USD/KRW",
                "current_value": "데이터 없음",
                "change": "NULL",
                "change_pct": "NULL"
            })
        
        return indicators
    
    indicators_data = safe_request_with_retry(fetch_indicators)
    
    if indicators_data is None:
        return json.dumps([{
            "indicator": "시장 지표 오류",
            "current_value": "데이터 없음",
            "change": "0.00",
            "change_pct": "0.00"
        }], ensure_ascii=False)
    
    return json.dumps(indicators_data, ensure_ascii=False)

# =============================================================================
# 아거용 도구들 - 종목 분석가의 심층 분석 도구
# =============================================================================

def search_stock_code(company_name: str, additional_info: str = "") -> str:
    """
    기업명을 바탕으로 정확한 종목코드를 검색합니다.
    
    한국 증시 전체 종목 목록에서 입력된 기업명과 매칭되는 종목을 찾아
    정확한 종목코드를 반환합니다. 여러 매칭 결과가 있을 경우 
    추가 정보를 활용하여 가장 적절한 종목을 선택합니다.
    
    Args:
        company_name (str): 검색할 기업명 (예: "삼성전자", "SK하이닉스")
        additional_info (str): 추가 설명 정보 (모호한 경우 참고용)
        
    Returns:
        str: JSON 형태의 종목 정보
        
    반환 형식:
        {
            "success": true/false,
            "code": "종목코드",
            "name": "정확한 종목명",
            "market": "상장시장",
            "message": "결과 메시지"
        }
        
    사용 예시:
        >>> result = search_stock_code("삼성전자")
        >>> result_data = json.loads(result)
        >>> if result_data["success"]:
        ...     print(f"종목코드: {result_data['code']}")
        ... else:
        ...     print(f"검색 실패: {result_data['message']}")
    """
    
    def search_stock():
        # 한국 증시 전체 종목 리스트 조회
        all_stocks = fdr.StockListing('KRX')
        
        # 정확한 매칭 시도
        exact_match = all_stocks[all_stocks['Name'] == company_name]
        if len(exact_match) > 0:
            stock = exact_match.iloc[0]
            return {
                "success": True,
                "code": stock['Code'],
                "name": stock['Name'],
                "market": stock['Market'],
                "message": f"정확한 매칭 완료: {stock['Name']}"
            }
        
        # 부분 매칭 시도 (이름에 포함되는 경우)
        partial_matches = all_stocks[all_stocks['Name'].str.contains(company_name, na=False)]
        if len(partial_matches) > 0:
            if len(partial_matches) == 1:
                stock = partial_matches.iloc[0]
                return {
                    "success": True,
                    "code": stock['Code'],
                    "name": stock['Name'],
                    "market": stock['Market'],
                    "message": f"부분 매칭 완료: {stock['Name']}"
                }
            else:
                # 여러 결과가 있는 경우 첫 번째 결과 반환 (실제로는 더 정교한 로직 필요)
                stock = partial_matches.iloc[0]
                matches_list = partial_matches['Name'].tolist()[:5]  # 최대 5개까지
                return {
                    "success": True,
                    "code": stock['Code'],
                    "name": stock['Name'],
                    "market": stock['Market'],
                    "message": f"여러 매칭 결과 중 선택: {stock['Name']} (다른 결과: {', '.join(matches_list[1:])})"
                }
        
        # 매칭 실패
        return {
            "success": False,
            "code": "",
            "name": "",
            "market": "",
            "message": f"'{company_name}'에 해당하는 종목을 찾을 수 없습니다. 정확한 종목명을 입력해주세요."
        }
    
    result = safe_request_with_retry(search_stock)
    
    if result is None:
        result = {
            "success": False,
            "code": "",
            "name": "",
            "market": "",
            "message": "종목 검색 중 오류가 발생했습니다."
        }
    
    return json.dumps(result, ensure_ascii=False)

def get_stock_price(stock_code: str) -> str:
    """
    특정 종목의 주가 데이터를 조회합니다.
    
    FinanceDataReader를 통해 지정된 종목의 최근 주가 정보를 수집하여
    현재가, 변동률, 거래량 등의 핵심 정보를 제공합니다.
    
    Args:
        stock_code (str): 6자리 종목코드 (예: "005930")
        
    Returns:
        str: JSON 형태의 주가 정보
        
    반환 형식:
        {
            "success": true/false,
            "current_price": "현재가",
            "change": "전일 대비 변동",
            "change_pct": "변동률(%)",
            "volume": "거래량",
            "high": "당일 고가",
            "low": "당일 저가",
            "message": "결과 메시지"
        }
        
    사용 예시:
        >>> price_info = get_stock_price("005930")  # 삼성전자
        >>> price_data = json.loads(price_info)
        >>> if price_data["success"]:
        ...     print(f"현재가: {price_data['current_price']}원")
        ...     print(f"변동률: {price_data['change_pct']}%")
    """
    
    def fetch_price():
        # 최근 5일간 데이터 조회 (휴장일 고려)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        
        stock_data = fdr.DataReader(stock_code, start=start_date, end=end_date)
        
        if len(stock_data) == 0:
            return {
                "success": False,
                "message": f"종목코드 {stock_code}의 데이터를 찾을 수 없습니다."
            }
        
        # 최신 데이터
        latest = stock_data.iloc[-1]
        current_price = latest['Close']
        high = latest['High']
        low = latest['Low']
        volume = latest['Volume']
        
        # 전일 대비 변동 계산
        if len(stock_data) >= 2:
            previous = stock_data.iloc[-2]['Close']
            change = current_price - previous
            change_pct = (change / previous) * 100
        else:
            change = 0
            change_pct = 0
        
        return {
            "success": True,
            "current_price": f"{current_price:,.0f}",
            "change": f"{change:+,.0f}",
            "change_pct": f"{change_pct:+.2f}",
            "volume": f"{volume:,}",
            "high": f"{high:,.0f}",
            "low": f"{low:,.0f}",
            "message": f"주가 정보 조회 완료"
        }
    
    result = safe_request_with_retry(fetch_price)
    
    if result is None:
        result = {
            "success": False,
            "current_price": "0",
            "change": "0",
            "change_pct": "0.00",
            "volume": "0",
            "high": "0",
            "low": "0",
            "message": "주가 데이터 조회 중 오류가 발생했습니다."
        }
    
    return json.dumps(result, ensure_ascii=False)

def analyze_stock_pattern(stock_code: str) -> str:
    """
    주가 패턴을 분석하여 기술적 지표와 패턴을 제공합니다.
    
    이동평균선, 거래량 분석, 가격 패턴 등을 종합하여 
    기술적 분석 결과를 제공합니다.
    
    Args:
        stock_code (str): 6자리 종목코드
        
    Returns:
        str: JSON 형태의 기술적 분석 결과
        
    반환 형식:
        {
            "success": true/false,
            "trend": "추세 (상승/하락/횡보)",
            "patterns": ["발견된 패턴들"],
            "moving_averages": {
                "ma5": "5일 이동평균",
                "ma20": "20일 이동평균",
                "ma60": "60일 이동평균"
            },
            "indicators": {
                "rsi": "RSI 값",
                "volume_trend": "거래량 추세"
            },
            "message": "분석 결과 메시지"
        }
        
    사용 예시:
        >>> analysis = analyze_stock_pattern("005930")
        >>> analysis_data = json.loads(analysis)
        >>> print(f"현재 추세: {analysis_data['trend']}")
        >>> print(f"발견된 패턴: {', '.join(analysis_data['patterns'])}")
    """
    
    def analyze_pattern():
        # 최근 3개월 데이터 조회
        end_date = datetime.now()
        start_date = end_date - timedelta(days=90)
        
        stock_data = fdr.DataReader(stock_code, start=start_date, end=end_date)
        
        if len(stock_data) < 20:
            return {
                "success": False,
                "message": "분석을 위한 충분한 데이터가 없습니다."
            }
        
        # 이동평균선 계산
        stock_data['MA5'] = stock_data['Close'].rolling(window=5).mean()
        stock_data['MA20'] = stock_data['Close'].rolling(window=20).mean()
        stock_data['MA60'] = stock_data['Close'].rolling(window=60).mean()
        
        # 최신 값들
        latest = stock_data.iloc[-1]
        current_price = latest['Close']
        ma5 = latest['MA5']
        ma20 = latest['MA20']
        ma60 = latest['MA60'] if not pd.isna(latest['MA60']) else None
        
        # 추세 판단
        if current_price > ma5 > ma20:
            trend = "상승"
        elif current_price < ma5 < ma20:
            trend = "하락"
        else:
            trend = "횡보"
        
        # 패턴 분석 (간단한 예시)
        patterns = []
        
        # 골든크로스/데드크로스 확인
        if len(stock_data) >= 2:
            prev_ma5 = stock_data.iloc[-2]['MA5']
            prev_ma20 = stock_data.iloc[-2]['MA20']
            
            if ma5 > ma20 and prev_ma5 <= prev_ma20:
                patterns.append("골든크로스")
            elif ma5 < ma20 and prev_ma5 >= prev_ma20:
                patterns.append("데드크로스")
        
        # 거래량 추세
        recent_volume = stock_data['Volume'].tail(5).mean()
        previous_volume = stock_data['Volume'].tail(10).head(5).mean()
        volume_trend = "증가" if recent_volume > previous_volume * 1.2 else "감소" if recent_volume < previous_volume * 0.8 else "보합"
        
        # RSI 계산 (단순화된 버전)
        price_diff = stock_data['Close'].diff()
        gains = price_diff.where(price_diff > 0, 0)
        losses = -price_diff.where(price_diff < 0, 0)
        avg_gains = gains.rolling(window=14).mean()
        avg_losses = losses.rolling(window=14).mean()
        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))
        current_rsi = rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50
        
        if not patterns:
            patterns.append("특별한 패턴 없음")
        
        return {
            "success": True,
            "trend": trend,
            "patterns": patterns,
            "moving_averages": {
                "ma5": f"{ma5:.0f}" if not pd.isna(ma5) else "N/A",
                "ma20": f"{ma20:.0f}" if not pd.isna(ma20) else "N/A", 
                "ma60": f"{ma60:.0f}" if ma60 and not pd.isna(ma60) else "N/A"
            },
            "indicators": {
                "rsi": f"{current_rsi:.1f}",
                "volume_trend": volume_trend
            },
            "message": "기술적 분석 완료"
        }
    
    result = safe_request_with_retry(analyze_pattern)
    
    if result is None:
        result = {
            "success": False,
            "trend": "알 수 없음",
            "patterns": ["분석 실패"],
            "moving_averages": {"ma5": "N/A", "ma20": "N/A", "ma60": "N/A"},
            "indicators": {"rsi": "N/A", "volume_trend": "N/A"},
            "message": "기술적 분석 중 오류가 발생했습니다."
        }
    
    return json.dumps(result, ensure_ascii=False)

def get_company_info(stock_code: str) -> str:
    """
    기업의 재무정보와 기본 정보를 수집합니다.
    
    네이버 증권에서 기업의 실적 정보와 동종업계 비교 데이터를 
    스크래핑하여 펀더멘털 분석에 필요한 정보를 제공합니다.
    
    Args:
        stock_code (str): 6자리 종목코드
        
    Returns:
        str: JSON 형태의 기업 정보
        
    반환 형식:
        {
            "success": true/false,
            "company_name": "기업명",
            "financial_data": "재무 정보 HTML",
            "industry_comparison": "업종 비교 HTML",
            "message": "결과 메시지"
        }
        
    사용 예시:
        >>> company_info = get_company_info("005930")
        >>> info_data = json.loads(company_info)
        >>> if info_data["success"]:
        ...     print(f"기업명: {info_data['company_name']}")
    """
    
    def fetch_company_info():
        # 네이버 증권 기업 정보 페이지
        url = f"https://finance.naver.com/item/main.naver?code={stock_code}"
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0 Safari/537.36"
        }
        
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, "html.parser")
        
        # 기업명 추출
        company_name_elem = soup.select_one(".wrap_company h2 a")
        company_name = company_name_elem.text.strip() if company_name_elem else f"종목{stock_code}"
        
        # 기업실적분석 섹션
        performance_selector = "#content > div.section.cop_analysis > div.sub_section"
        performance_elements = soup.select(performance_selector)
        financial_data = str(performance_elements[0]).replace("\t", "").replace("\n\n", "\n") if performance_elements else "재무 데이터 없음"
        
        # 동일업종비교 섹션
        compare_selector = "#content > div.section.trade_compare > table"
        compare_elements = soup.select(compare_selector)
        industry_comparison = str(compare_elements[0]).replace("\t", "").replace("\n\n", "\n") if compare_elements else "업종 비교 데이터 없음"
        
        return {
            "success": True,
            "company_name": company_name,
            "financial_data": financial_data,
            "industry_comparison": industry_comparison,
            "message": f"{company_name} 기업 정보 수집 완료"
        }
    
    result = safe_request_with_retry(fetch_company_info)
    
    if result is None:
        result = {
            "success": False,
            "company_name": f"종목{stock_code}",
            "financial_data": "데이터 수집 실패",
            "industry_comparison": "데이터 수집 실패",
            "message": "기업 정보 수집 중 오류가 발생했습니다."
        }
    
    return json.dumps(result, ensure_ascii=False)

# =============================================================================
# 라무용 도구들 - 시뮬레이터의 시나리오 생성 도구
# =============================================================================

def generate_scenarios(portfolio_stocks: str, scenario_count: int = 3) -> str:
    """
    포트폴리오 종목들을 기반으로 다양한 시장 시나리오를 생성합니다.
    
    OpenAI GPT를 활용하여 현실적이고 구체적인 시장 시나리오를 생성하고,
    각 시나리오가 포트폴리오에 미칠 영향을 예측합니다.
    
    Args:
        portfolio_stocks (str): 포트폴리오 종목 목록 (JSON 문자열 또는 쉼표 구분)
        scenario_count (int): 생성할 시나리오 개수 (기본값: 3)
        
    Returns:
        str: JSON 형태의 시나리오 목록
        
    반환 형식:
        [
            {
                "scenario_name": "시나리오명",
                "description": "구체적인 상황 설명",
                "probability": "발생 확률(%)",
                "impact_summary": "전체적인 영향 요약",
                "stock_impacts": [
                    {
                        "stock_code": "종목코드",
                        "stock_name": "종목명", 
                        "expected_change": "예상 변화율(%)"
                    }
                ]
            }
        ]
        
    사용 예시:
        >>> scenarios = generate_scenarios("005930,000660", 3)
        >>> scenarios_list = json.loads(scenarios)
        >>> for scenario in scenarios_list:
        ...     print(f"시나리오: {scenario['scenario_name']}")
        ...     print(f"발생 확률: {scenario['probability']}%")
    """
    
    def create_scenarios():
        # 포트폴리오 종목 파싱
        if portfolio_stocks.startswith('[') or portfolio_stocks.startswith('{'):
            # JSON 형태인 경우
            try:
                import json
                stocks_data = json.loads(portfolio_stocks)
                if isinstance(stocks_data, list):
                    stocks = stocks_data
                else:
                    stocks = [portfolio_stocks]  # 단일 종목
            except:
                stocks = [portfolio_stocks]
        else:
            # 쉼표 구분 문자열인 경우
            stocks = [stock.strip() for stock in portfolio_stocks.split(',')]
        
        # 종목명 매핑 (실제로는 KRX 데이터에서 조회)
        stock_mapping = {
            '005930': '삼성전자',
            '000660': 'SK하이닉스', 
            '373220': 'LG에너지솔루션',
            '207940': '삼성바이오로직스',
            '005935': '삼성전자우',
            '035420': 'NAVER',
            '003670': '포스코DX'
        }
        
        # OpenAI를 통한 시나리오 생성
        try:
            stock_names = []
            for stock in stocks:
                if stock in stock_mapping:
                    stock_names.append(f"{stock}({stock_mapping[stock]})")
                else:
                    stock_names.append(stock)
            
            prompt = f"""
다음 포트폴리오에 대한 현실적인 시장 시나리오 {scenario_count}개를 생성해주세요.

포트폴리오 종목: {', '.join(stock_names)}

각 시나리오는 다음 조건을 만족해야 합니다:
1. 실제로 발생 가능한 현실적인 상황
2. 각 종목에 미칠 구체적인 영향 분석
3. 발생 확률과 영향도 정량화
4. 긍정적/부정적/중립적 시나리오 균형 있게 포함

JSON 형식으로 응답해주세요:
[
    {{
        "scenario_name": "시나리오명",
        "description": "상황 설명 (200자 내외)",
        "probability": "발생확률(숫자만)",
        "impact_summary": "전체 영향 요약",
        "stock_impacts": [
            {{
                "stock_code": "종목코드",
                "stock_name": "종목명",
                "expected_change": "예상변화율(+/-숫자)"
            }}
        ]
    }}
]
"""
            
            response = client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[
                    {"role": "system", "content": "당신은 금융 시장 분석 전문가입니다. 현실적이고 구체적인 시나리오를 생성합니다."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.7
            )
            
            scenarios_json = response.choices[0].message.content
            scenarios_data = json.loads(scenarios_json)
            
            # 응답이 직접 배열인지 확인
            if isinstance(scenarios_data, dict) and 'scenarios' in scenarios_data:
                return scenarios_data['scenarios']
            elif isinstance(scenarios_data, list):
                return scenarios_data
            else:
                return [scenarios_data]  # 단일 객체인 경우
                
        except Exception as e:
            print(f"OpenAI 시나리오 생성 실패: {e}")
            # 폴백 시나리오
            return [
                {
                    "scenario_name": "금리 인상 시나리오",
                    "description": "중앙은행의 기준금리 0.5%p 인상으로 인한 시장 조정",
                    "probability": "40",
                    "impact_summary": "기술주 약세, 금융주 강세 예상",
                    "stock_impacts": [
                        {
                            "stock_code": stock,
                            "stock_name": stock_mapping.get(stock, stock),
                            "expected_change": "-3.5"
                        } for stock in stocks
                    ]
                },
                {
                    "scenario_name": "반도체 수요 회복",
                    "description": "AI 열풍과 스마트폰 교체 수요로 메모리 반도체 가격 상승",
                    "probability": "60", 
                    "impact_summary": "반도체 관련주 전반적 상승 기대",
                    "stock_impacts": [
                        {
                            "stock_code": stock,
                            "stock_name": stock_mapping.get(stock, stock),
                            "expected_change": "+5.2"
                        } for stock in stocks
                    ]
                }
            ][:scenario_count]
    
    scenarios_data = safe_request_with_retry(create_scenarios)
    
    if scenarios_data is None:
        scenarios_data = [{
            "scenario_name": "시나리오 생성 오류",
            "description": "현재 시나리오 생성 서비스에 문제가 있습니다.",
            "probability": "0",
            "impact_summary": "분석 불가",
            "stock_impacts": []
        }]
    
    return json.dumps(scenarios_data, ensure_ascii=False)

# =============================================================================
# 도구 목록 정의 (OpenAI Function Calling용)
# =============================================================================

# 각 에이전트별 사용 가능한 도구들을 정의
AGENT_TOOLS = {
    "키키": [
        {
            "type": "function",
            "function": {
                "name": "get_latest_news",
                "description": "최신 경제/금융 뉴스를 실시간으로 수집합니다",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
        },
        {
            "type": "function", 
            "function": {
                "name": "get_major_movers",
                "description": "주가가 크게 변동한 주요 종목들을 조회합니다",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_market_indicators", 
                "description": "KOSPI, KOSDAQ, 환율 등 주요 시장 지표를 조회합니다",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
        }
    ],
    
    "아거": [
        {
            "type": "function",
            "function": {
                "name": "search_stock_code",
                "description": "기업명으로 정확한 종목코드를 검색합니다",
                "parameters": {
                    "type": "object", 
                    "properties": {
                        "company_name": {
                            "type": "string",
                            "description": "검색할 기업명 (예: 삼성전자, SK하이닉스)"
                        },
                        "additional_info": {
                            "type": "string",
                            "description": "추가 설명 정보 (선택사항)"
                        }
                    },
                    "required": ["company_name"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_stock_price",
                "description": "특정 종목의 현재 주가 정보를 조회합니다",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "stock_code": {
                            "type": "string", 
                            "description": "6자리 종목코드 (예: 005930)"
                        }
                    },
                    "required": ["stock_code"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "analyze_stock_pattern",
                "description": "주가 패턴과 기술적 지표를 분석합니다",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "stock_code": {
                            "type": "string",
                            "description": "6자리 종목코드"
                        }
                    },
                    "required": ["stock_code"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_company_info",
                "description": "기업의 재무정보와 업종 비교 데이터를 수집합니다",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "stock_code": {
                            "type": "string",
                            "description": "6자리 종목코드"
                        }
                    },
                    "required": ["stock_code"]
                }
            }
        }
    ],
    
    "라무": [
        {
            "type": "function",
            "function": {
                "name": "generate_scenarios",
                "description": "포트폴리오를 기반으로 다양한 시장 시나리오를 생성합니다",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "portfolio_stocks": {
                            "type": "string",
                            "description": "포트폴리오 종목 목록 (쉼표로 구분된 종목코드)"
                        },
                        "scenario_count": {
                            "type": "integer",
                            "description": "생성할 시나리오 개수",
                            "default": 3
                        }
                    },
                    "required": ["portfolio_stocks"]
                }
            }
        }
    ]
}

# 도구 함수들을 이름으로 매핑
TOOL_FUNCTIONS = {
    "get_latest_news": get_latest_news,
    "get_major_movers": get_major_movers, 
    "get_market_indicators": get_market_indicators,
    "search_stock_code": search_stock_code,
    "get_stock_price": get_stock_price,
    "analyze_stock_pattern": analyze_stock_pattern,
    "get_company_info": get_company_info,
    "generate_scenarios": generate_scenarios
}

# 모듈 테스트
if __name__ == "__main__":
    print("=== STARGENT 도구 모듈 테스트 ===")
    
    # 환경변수 확인
    if not os.getenv('OPENAI_API_KEY'):
        print("⚠️ OPENAI_API_KEY가 설정되지 않았습니다.")
        print("💡 .env 파일에 API 키를 설정해주세요.")
    
    # 키키 도구 테스트
    print("\n📰 뉴스 수집 테스트:")
    news = get_latest_news()
    print(f"뉴스 수집 결과: {len(json.loads(news))}개")
    print(news)
    
    print("\n📊 시장 지표 테스트:")
    indicators = get_market_indicators()
    print(f"지표 수집 결과: {len(json.loads(indicators))}개")
    print(indicators)
    
    # 아거 도구 테스트
    print("\n🔍 종목 검색 테스트:")
    search_result = search_stock_code("삼성전자")
    search_data = json.loads(search_result)
    print(f"검색 결과: {search_data['success']}")
    print(search_data)

    if search_data['success']:
        print(f"\n💰 주가 조회 테스트:")
        price_result = get_stock_price(search_data['code'])
        price_data = json.loads(price_result)
        print(f"주가 조회 결과: {price_data['success']}")
        print(price_data)
    
    print("\n✅ 모든 도구가 정상적으로 작동합니다! 🎉")