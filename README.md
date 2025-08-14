
# STARGENT - KB 금융 AI 에이전트 데모

STARGENT는 KB 금융그룹의 스타프렌즈 캐릭터를 활용한 멀티 AI 에이전트 기반 금융 서비스 데모입니다. 5명의 전문 에이전트가 협업하여 사용자에게 개인화된 금융 상담과 투자 조언을 제공합니다.

> 해당 리포지토리의 코드는 STARGENT 시스템의 일부 기능을 구현한 Streamlit 기반 데모 코드입니다.

## 개발 디렉토리 구조

```
stargent-demo/
├── app/                        # 애플리케이션 메인 코드
│   ├── agent/                  # AI 에이전트 관련 모듈
│   │   ├── model_class.py      # 데이터 모델 정의
│   │   ├── tools.py           # 에이전트가 사용할 도구들
│   │   └── agent.py           # 에이전트 실행 로직
│   ├── streamlit_app.py       # Streamlit UI 메인 파일
│   └── load_dotenv.py         # 환경변수 로딩 유틸리티
├── prompts/                    # 에이전트별 프롬프트 파일들
│   ├── manager.md
│   ├── bibi.md
│   ├── kiki.md
│   ├── ager.md
│   ├── ramu.md
│   └── coli.md
├── .env                       # 실제 환경변수 (gitignore)
├── .env.example              # 환경변수 템플릿
├── requirements.txt          # Python 의존성
├── Dockerfile               # Docker 배포 설정
└── README.md               # 프로젝트 문서
```

## 실행 방법

### 로컬 개발 환경

```bash
# 환경변수로 openai api 키 설정하기 (자동으로 .env파일에서 로드되지 않는 경우, 윈도우 기준)
$Env:OPENAI_API_KEY = "your_key"

# 실행
streamlit run app/streamlit_app.py --server.port 8501
```

### Docker

```bash
# 이미지 빌드
docker build -t stargent-demo .

# 실행
docker run -d -p 8501:8501 -e OPENAI_API_KEY=your_key --name stargent-container stargent-demo

```

- 실행 후: `localhost:8501`로 접속하세요  

## 에이전트 시스템

### 데이터 모델 설계

```python
@dataclass
class UserContexts:
    """사용자 정보와 채팅 기록 관리"""
    user_info: Dict[str, Any]      # 투자 성향, 포트폴리오 등
    chat_history: List[Dict]       # 전체 대화 기록

@dataclass  
class AgentContexts:
    """에이전트 작업 상태 관리"""
    agent_id_history: List[str]    # 실행된 에이전트 순서
    total_step: int                # 전체 작업 단계
    current_step: int              # 현재 진행 단계
    agent_output: List[AgentOutput] # 각 에이전트 결과
```

### 도구 시스템 (Tools)

#### 키키 (트렌드 마스터) 도구

- `get_latest_news()`: 네이버 금융 뉴스 실시간 수집
- `get_major_movers()`: 급등/급락 종목 조회
- `get_market_indicators()`: KOSPI/KOSDAQ/환율 등 주요 지표

#### 아거 (종목 분석가) 도구

- `search_stock_code()`: 기업명으로 종목코드 검색
- `get_stock_price()`: 실시간 주가 정보 조회
- `analyze_stock_pattern()`: 기술적 분석 (이동평균, RSI 등)
- `get_company_info()`: 기업 정보 및 재무 데이터

#### 라무 (시뮬레이터) 도구

- `generate_scenarios()`: OpenAI 기반 시나리오 생성

### 워크플로우 처리

1. **사용자 입력** → 매니저 에이전트가 요청 분석
2. **PLAN 수립** → 작업 단계와 담당 에이전트 결정
3. **순차 실행** → 각 에이전트가 전문 영역 작업 수행
4. **결과 통합** → 최종 응답을 사용자에게 전달


### 프롬프트

각 에이전트의 프롬프트는 `prompts/` 디렉토리의 마크다운 파일에 위치합니다.

- `manager.md`: 매니저 에이전트
- `bibi.md`: 비비 (대화 친구)
- `kiki.md`: 키키 (트렌드 마스터)
- `ager.md`: 아거 (종목 분석가)
- `ramu.md`: 라무 (시뮬레이터)
- `coli.md`: 콜리 (펀드 매니저)

