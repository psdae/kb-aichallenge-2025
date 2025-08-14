"""
STARGENT 데모 서비스 메인 애플리케이션

이 파일은 Streamlit을 사용하여 STARGENT AI 에이전트 시스템의
사용자 인터페이스를 구현합니다. 개선된 UI/UX로 채팅 중심의 
직관적인 사용자 경험을 제공합니다.

주요 개선사항:
    1. 심플하고 미려한 디자인
    2. 채팅 인터페이스 우선 배치
    3. 설명 섹션을 접이식으로 정리
    4. 과도한 이모지 제거
    5. 사용자 정보 편집 인터페이스 개선
"""

import streamlit as st
import json
import time
import os
from datetime import datetime
from pathlib import Path

# 프로젝트 모듈 임포트
from load_dotenv import load_env, env_load_test
from agent.model_class import UserContexts, AgentContexts, AgentOutput, AgentPlan
from agent.agent import execute_manager, execute_plan, AGENT_EXECUTORS

# 환경변수 로드
load_env()

# =============================================================================
# Streamlit 페이지 설정 및 스타일링
# =============================================================================

def setup_page_config():
    """Streamlit 페이지 기본 설정을 구성합니다."""
    st.set_page_config(
        page_title="STARGENT - KB 금융 AI 에이전트",
        page_icon="🌟",
        layout="centered",
        initial_sidebar_state="expanded",
        menu_items={
            'About': "STARGENT는 KB 금융그룹의 AI 에이전트 기반 금융 서비스입니다."
        }
    )

    # 라이트 테마 강제 적용을 위한 설정
    try:
        import streamlit.config as config
        config.set_option('theme.base', 'light')
        config.set_option('theme.primaryColor', '#FFCC00')
        config.set_option('theme.backgroundColor', '#FFFFFF')
        config.set_option('theme.secondaryBackgroundColor', '#F8F9FA')
        config.set_option('theme.textColor', '#2C3E50')
    except:
        pass  # 설정이 실패해도 앱 실행에는 영향 없음
    
    # 심플하고 깔끔한 CSS 스타일
    st.markdown("""
    <style>
    /* 기본 Streamlit 요소 색상 변경 */
    .stButton > button {
        background-color: #FFCC00;
        color: #2C3E50;
        border: none;
        border-radius: 6px;
        font-weight: 500;
        transition: background-color 0.3s ease;
    }
    
    .stButton > button:hover {
        background-color: #E6B800;
        color: #2C3E50;
    }
    
    /* 탭 스타일링 */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #f8f9fa;
        border-radius: 6px 6px 0 0;
        color: #2C3E50;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #FFCC00;
        color: #2C3E50;
    }
    
    /* 프로그레스 바 색상 */
    .stProgress > div > div > div {
        background-color: #FFCC00;
    }
    
    /* 사이드바 헤더 스타일 */
    .sidebar-header {
        background-color: #FFCC00;
        color: #2C3E50;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    
    /* 채팅 메시지 스타일링 개선 */
    .stChatMessage {
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    
    /* 간단한 카드 스타일 */
    .info-card {
        background: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    /* 채팅 입력 영역 강조 */
    .stChatInput {
        border: 2px solid #FFCC00;
        border-radius: 8px;
    }
    
    /* 모바일 반응형 */
    @media (max-width: 768px) {
        .sidebar-header {
            padding: 0.5rem;
            font-size: 0.9rem;
        }
    }
    </style>
    """, unsafe_allow_html=True)

def initialize_session_state():
    """Streamlit 세션 상태를 초기화합니다."""
    # 사용자 컨텍스트 초기화
    if "user_contexts" not in st.session_state:
        st.session_state.user_contexts = UserContexts(
            user_info={
                "name": "김투자",
                "age": "35",
                "investment_style": "중도적 투자자",
                "experience": "3년",
                "portfolio": "삼성전자 50%, SK하이닉스 30%, NAVER 20%",
                "monthly_investment": "100만원",
                "risk_tolerance": "중간 위험 허용",
                "goals": "안정적인 장기 투자로 은퇴 자금 마련"
            }
        )
    
    # 에이전트 컨텍스트 초기화
    if "agent_contexts" not in st.session_state:
        st.session_state.agent_contexts = AgentContexts()
    
    # 채팅 기록이 없으면 웰컴 메시지 추가
    if not st.session_state.user_contexts.chat_history:
        welcome_message = """안녕하세요! 저는 KB 스타프렌즈의 비비입니다.

오늘도 투자와 금융에 대한 궁금한 것들이 많으시죠? 
저와 함께 스타프렌즈 친구들이 도와드릴게요!

• **키키**: 최신 시장 뉴스와 트렌드 분석

• **아거**: 종목 심층 분석과 기업 정보

• **라무**: 포트폴리오 시뮬레이션과 시나리오 분석  

• **콜리**: 투자 전략과 포트폴리오 관리

어떤 도움이 필요하신지 편하게 말씀해주세요!"""
        st.session_state.user_contexts.add_assistant_message(welcome_message)
    
    # 기타 상태 초기화
    if "is_processing" not in st.session_state:
        st.session_state.is_processing = False

# =============================================================================
# 사이드바 - 사용자 정보 관리
# =============================================================================

def render_sidebar():
    """사이드바에 사용자 정보 편집 인터페이스를 렌더링합니다."""
    with st.sidebar:
        # 헤더
        st.markdown("""
        <div class="sidebar-header">
            STARGENT<br>
            <small>KB 금융 AI 에이전트</small>
        </div>
        """, unsafe_allow_html=True)
        
        st.subheader("사용자 정보")
        
        user_info = st.session_state.user_contexts.user_info
        
        # 편집 가능한 사용자 정보 입력 폼
        with st.form("user_info_form"):
            new_name = st.text_input("이름", value=user_info.get('name', ''))
            new_age = st.text_input("나이", value=user_info.get('age', ''))
            new_investment_style = st.selectbox(
                "투자 성향", 
                options=["보수적 투자자", "중도적 투자자", "공격적 투자자"],
                index=["보수적 투자자", "중도적 투자자", "공격적 투자자"].index(
                    user_info.get('investment_style', '중도적 투자자')
                )
            )
            new_experience = st.text_input("투자 경험", value=user_info.get('experience', ''))
            new_portfolio = st.text_area("현재 포트폴리오", value=user_info.get('portfolio', ''))
            new_monthly_investment = st.text_input("월 투자 금액", value=user_info.get('monthly_investment', ''))
            new_risk_tolerance = st.selectbox(
                "위험 허용도",
                options=["낮은 위험 허용", "중간 위험 허용", "높은 위험 허용"],
                index=["낮은 위험 허용", "중간 위험 허용", "높은 위험 허용"].index(
                    user_info.get('risk_tolerance', '중간 위험 허용')
                )
            )
            new_goals = st.text_area("투자 목표", value=user_info.get('goals', ''))
            
            # 업데이트 버튼
            if st.form_submit_button("정보 업데이트하기", use_container_width=True):
                # 사용자 정보 업데이트
                st.session_state.user_contexts.user_info = {
                    'name': new_name,
                    'age': new_age,
                    'investment_style': new_investment_style,
                    'experience': new_experience,
                    'portfolio': new_portfolio,
                    'monthly_investment': new_monthly_investment,
                    'risk_tolerance': new_risk_tolerance,
                    'goals': new_goals
                }
                st.success("사용자 정보가 업데이트되었습니다!")
                st.rerun()
        
        # st.markdown("---")
        
        # 채팅 기록 초기화 버튼
        if st.button("채팅 기록 초기화", type="secondary", use_container_width=True):
            st.session_state.user_contexts.chat_history = []
            initialize_session_state()
            st.success("채팅 기록이 초기화되었습니다!")
            st.rerun()

        # 푸터
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; opacity: 0.6; padding: 0.5rem;">
            <small>STARGENT v1.0 | KB 금융 AI 공모전 출품작 | Powered by OpenAI gpt-4.1-mini & Streamlit</small>
        </div>
        """, unsafe_allow_html=True)

# =============================================================================
# 메인 채팅 인터페이스
# =============================================================================

def render_starter_questions():
    """채팅 사용 예시를 제공합니다."""
    st.write("**채팅 사용 예시:**")
    
    examples = [
        "오늘 시장 상황은 어때?",
        "오늘 시장 뉴스를 정리해줘",
        "오늘의 시황을 정리해줘",
        "내 투자 포트폴리오를 분석하고 앞으로 어떻게 투자해야 할 지 추천해줘",
        "삼성전자의 기업 정보를 알려줘",
        "하이닉스의 기업 정보를 알려줘",
        "앞으로 어떻게 투자해야 할 지 시나리오별로 분석해볼래?"
    ]
    
    for example in examples:
        st.write(f"• {example}")
    
    # st.write("위 예시를 참고하여 자유롭게 질문해보세요!")


def process_user_message(user_input: str):
    """사용자 메시지를 처리하여 적절한 에이전트를 실행합니다."""
    if not user_input.strip():
        return
    
    # API 키 확인
    if not os.getenv('OPENAI_API_KEY'):
        st.error("OpenAI API 키가 설정되지 않았습니다. .env 파일을 확인해주세요.")
        return
    
    # 사용자 메시지 추가
    st.session_state.user_contexts.add_user_message(user_input)
    
    # 처리 중 상태 설정
    st.session_state.is_processing = True
    
    # 진행상황 표시 컨테이너
    progress_container = st.empty()
    status_container = st.empty()
    
    try:
        with status_container.container():
            with st.status("매니저가 작업 계획을 수립 중입니다...", expanded=True) as status:
                st.write("사용자 요청을 분석하고 있습니다...")
                
                # 1. 매니저 실행 - 작업 계획 수립
                plan = execute_manager(st.session_state.user_contexts, st.session_state.agent_contexts)
                
                st.write(f"작업 계획 수립 완료: {plan.total_steps}단계")
                for i, step in enumerate(plan.plans):
                    st.write(f"{i+1}. {step['agent_name']}: {step['description']}")
                
                status.update(label="작업 계획 수립 완료!", state="complete")
        
        # 2. 계획 실행
        agent_ctx = AgentContexts(
            total_step=plan.total_steps,
            current_step=0,
            agent_id_history=[],
            agent_output=[]
        )
        
        # 각 단계별 실행
        for step_idx, step_plan in enumerate(plan.plans):
            agent_name = step_plan["agent_name"]
            description = step_plan["description"]
            
            with status_container.container():
                with st.status(f"{agent_name}가 작업 중입니다...", expanded=True) as status:
                    st.write(f"단계 {step_idx + 1}/{plan.total_steps}: {description}")
                    
                    # 진행률 표시
                    progress = (step_idx + 1) / plan.total_steps
                    progress_container.progress(progress, f"전체 진행률: {progress:.0%}")
                    
                    # 에이전트 실행
                    if agent_name in AGENT_EXECUTORS:
                        result = AGENT_EXECUTORS[agent_name](st.session_state.user_contexts, agent_ctx)
                    else:
                        # 알 수 없는 에이전트인 경우 비비로 대체
                        result = AGENT_EXECUTORS["비비"](st.session_state.user_contexts, agent_ctx)
                    
                    # 결과를 컨텍스트에 추가
                    agent_ctx.add_agent_result(agent_name, result)
                    
                    st.write(f"결과: {result.progress_description}")
                    if result.output:
                        preview = result.output[:200] + "..." if len(result.output) > 200 else result.output
                        st.write(f"응답 미리보기: {preview}")
                    
                    status.update(label=f"{agent_name} 작업 완료!", state="complete")
        
        # 3. 최종 결과 처리
        if agent_ctx.agent_output:
            final_result = agent_ctx.agent_output[-1]
            st.session_state.user_contexts.add_assistant_message(
                final_result.output,
                progress=f"전체 작업 완료 ({plan.total_steps}단계)"
            )
        else:
            st.session_state.user_contexts.add_assistant_message(
                "죄송합니다. 요청을 처리하는 중에 문제가 발생했습니다.",
                progress="처리 실패"
            )
        
        # 성공 메시지
        with status_container.container():
            st.success("모든 작업이 완료되었습니다!")
            
    except Exception as e:
        st.error(f"처리 중 오류가 발생했습니다: {str(e)}")
        st.session_state.user_contexts.add_assistant_message(
            "죄송합니다. 처리 중 오류가 발생했습니다. 다시 시도해주세요.",
            progress="오류 발생"
        )
    
    finally:
        # 처리 완료 상태로 변경
        st.session_state.is_processing = False
        progress_container.empty()
        status_container.empty()
        st.rerun()

def render_chat_interface():
    """메인 채팅 인터페이스를 렌더링합니다."""
    st.header("KB STARGENT Chat")


    # 빠른 시작 질문들 (접이식)
    with st.expander("채팅 예시 (이런 질문으로 시작해보세요)", expanded=False):
        render_starter_questions()

    # 에이전트 소개 (접이식)
    with st.expander("AI 에이전트 소개", expanded=False):
        st.write("**KB 스타프렌즈 에이전트들**")
        
        agents_info = [
            ("비비 (ESFJ)", "대화 친구 & 커뮤니케이션 매니저", "친근한 대화와 감정적 지지를 제공하며, 다른 전문 에이전트들과 연결해드려요"),
            ("키키 (ENFP)", "트렌드 마스터", "최신 시장 뉴스와 트렌드를 분석하여 투자 기회를 빠르게 포착해요"),
            ("아거 (ENTP)", "종목 분석가", "특정 종목을 심층 분석하여 투자 인사이트를 제공해요"),
            ("라무 (ISFP)", "시뮬레이터", "다양한 시나리오를 시뮬레이션하여 포트폴리오 리스크를 분석해요"),
            ("콜리 (ISTJ)", "펀드 매니저", "체계적인 포트폴리오 관리와 투자 전략을 제안해요")
        ]
        
        for name, role, description in agents_info:
            st.markdown(f"""
            <div class="info-card">
                <strong>{name}</strong> - {role}<br>
                <small>{description}</small>
            </div>
            """, unsafe_allow_html=True)
    
    # 채팅 기록 표시
    chat_container = st.container()
    
    with chat_container:
        for chat in st.session_state.user_contexts.chat_history:
            role = chat['role']
            content = chat['content']
            
            if role == 'user':
                with st.chat_message("user"):
                    st.markdown(content)
            else:
                with st.chat_message("assistant"):
                    st.markdown(content)
                    
                    # 진행상황이 있는 경우 표시
                    if chat.get('progress'):
                        st.caption(f"진행상황: {chat['progress']}")
    
    # 사용자 입력 섹션
    # 처리 중일 때는 입력 비활성화
    input_disabled = st.session_state.is_processing

    if not st.session_state.is_processing:
        user_input = st.chat_input(
            "궁금한 것을 물어보세요! (eg. 오늘 시장 상황은 어때?, 삼성전자의 기업 정보를 알려줘 등)",
            disabled=input_disabled
        )
    
    if user_input and not input_disabled:
        process_user_message(user_input)
    
    # 처리 중일 때 안내 메시지
    if st.session_state.is_processing:
        st.info("AI 에이전트들이 열심히 작업 중입니다... 잠시만 기다려주세요!")


# =============================================================================
# 팟캐스트 데모 인터페이스  
# =============================================================================

def render_podcast_interface():
    """팟캐스트 데모 인터페이스를 렌더링합니다."""
    st.header("팟캐스트 데모")
    
    # 간단한 팟캐스트 데모 페이지
    st.info("팟캐스트 기능은 개발 중입니다.")
    
    # 빈 채팅 형태의 인터페이스 구성
    st.subheader("AI 에이전트들의 대화")
    
    # 샘플 메시지
    with st.chat_message("assistant", avatar="🐻"):
        st.markdown("**비비**: 안녕하세요! 오늘은 AI 관련 투자에 대해 이야기해볼까요?")
    
    with st.chat_message("assistant", avatar="🎯"):
        st.markdown("**키키**: 좋은 주제네요! 최근 AI 관련주들이 정말 주목받고 있어요.")
    
    with st.chat_message("assistant", avatar="🦆"):
        st.markdown("**아거**: 데이터를 보니 흥미로운 패턴들이 보이는데요...")
    
    # 팟캐스트 컨트롤 (향후 구현 예정)
    st.markdown("""
    <div class="info-card">
        <p><strong>향후 구현 예정 기능:</strong></p>
        <ul>
            <li>에이전트들의 실시간 대화 생성</li>
            <li>음성 합성을 통한 실제 팟캐스트 재생</li>
            <li>주제별 맞춤 팟캐스트 생성</li>
            <li>사용자 맞춤형 금융 정보 팟캐스트</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # 더미 입력창 (비활성화됨)
    st.chat_input("팟캐스트 기능은 아직 구현되지 않았습니다.", disabled=True)

# =============================================================================
# 메인 애플리케이션
# =============================================================================

def main():
    """메인 애플리케이션 함수"""
    # 페이지 설정
    setup_page_config()
    
    # 세션 상태 초기화
    initialize_session_state()
    
    # 사이드바 렌더링
    render_sidebar()
    
    # # 메인 컨텐츠 영역 - 탭으로 구성
    # tab1, tab2 = st.tabs(["채팅", "팟캐스트"])
    
    # with tab1:
    #     render_chat_interface()
    
    # with tab2:
    #     render_podcast_interface()
    render_chat_interface()
    
    # # 푸터
    # st.markdown("---")
    # st.markdown("""
    # <div style="text-align: center; opacity: 0.6; padding: 0.5rem;">
    #     <small>STARGENT v1.0 | KB 금융 AI 공모전 출품작 | Powered by OpenAI gpt-4.1-mini & Streamlit</small>
    # </div>
    # """, unsafe_allow_html=True)


# 환경변수 로드
env_load_test()

if __name__ == "__main__":
    main()