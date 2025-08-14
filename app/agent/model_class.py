"""
STARGENT 시스템의 핵심 데이터 모델 정의

이 모듈은 AI 에이전트 시스템에서 사용되는 모든 데이터 구조를 정의합니다.
각 클래스는 특정한 역할을 가지며, JSON 직렬화를 지원하여 
세션 상태 관리와 데이터 교환을 용이하게 합니다.

주요 클래스:
    - UserContexts: 사용자 정보와 채팅 기록 관리
    - AgentContexts: 에이전트 작업 상태와 결과 관리  
    - AgentOutput: 개별 에이전트의 출력 표준화
    - AgentPlan: 매니저가 수립하는 작업 계획 구조
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
import json
from datetime import datetime

@dataclass
class UserContexts:
    """
    사용자 관련 정보와 대화 기록을 관리하는 클래스
    
    이 클래스는 개인화된 서비스 제공을 위해 사용자의 기본 정보와
    전체 채팅 히스토리를 체계적으로 관리합니다.
    
    Attributes:
        user_info (Dict[str, Any]): 사용자 기본 정보
            - name: 이름
            - age: 나이  
            - investment_style: 투자 성향 (보수적/중도적/공격적)
            - portfolio: 현재 보유 포트폴리오
            - risk_tolerance: 위험 허용도
        chat_history (List[Dict[str, Any]]): 채팅 기록
            각 채팅은 다음 구조를 가집니다:
            - role: 역할 ('user' 또는 'assistant')
            - content: 채팅 내용
            - timestamp: 시간 정보
            - progress: 진행상황 표시 (에이전트 응답용)
    
    사용 예시:
        >>> user_ctx = UserContexts(
        ...     user_info={'name': '김투자', 'age': 30, 'investment_style': '중도적'},
        ...     chat_history=[]
        ... )
        >>> user_ctx.add_user_message("삼성전자 분석해줘")
        >>> print(len(user_ctx.chat_history))
        1
    """
    
    user_info: Dict[str, Any] = field(default_factory=dict)
    chat_history: List[Dict[str, Any]] = field(default_factory=list)
    
    def add_user_message(self, content: str) -> None:
        """
        사용자 메시지를 채팅 기록에 추가합니다.
        
        Args:
            content (str): 사용자가 입력한 메시지
            
        사용 예시:
            >>> user_ctx.add_user_message("오늘 시장 상황은 어때?")
        """
        message = {
            'role': 'user',
            'content': content,
            'timestamp': datetime.now().isoformat(),
            'progress': None
        }
        self.chat_history.append(message)
    
    def add_assistant_message(self, content: str, progress: Optional[str] = None) -> None:
        """
        AI 어시스턴트 메시지를 채팅 기록에 추가합니다.
        
        Args:
            content (str): AI가 생성한 응답 내용
            progress (Optional[str]): 작업 진행상황 설명
            
        사용 예시:
            >>> user_ctx.add_assistant_message(
            ...     "분석 결과입니다...", 
            ...     progress="삼성전자 주가 데이터 분석 완료"
            ... )
        """
        message = {
            'role': 'assistant', 
            'content': content,
            'timestamp': datetime.now().isoformat(),
            'progress': progress
        }
        self.chat_history.append(message)
    
    def to_json(self) -> str:
        """객체를 JSON 문자열로 직렬화합니다."""
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'UserContexts':
        """JSON 문자열에서 객체를 역직렬화합니다."""
        data = json.loads(json_str)
        return cls(**data)

@dataclass 
class AgentOutput:
    """
    개별 에이전트의 출력을 표준화하는 클래스
    
    모든 에이전트는 작업 완료 후 이 형식으로 결과를 반환합니다.
    이를 통해 일관된 데이터 교환과 진행상황 추적이 가능합니다.
    
    Attributes:
        final_output (bool): 최종 출력 여부
            - True: 사용자에게 직접 보여줄 최종 결과
            - False: 다른 에이전트에게 전달할 중간 결과
        progress_description (str): 진행상황 설명
            사용자에게 현재 어떤 작업을 하고 있는지 알려주는 간단한 설명
        output (str): 실제 출력 내용
            - 중간 단계: 다음 에이전트를 위한 상세한 작업 결과
            - 최종 단계: 사용자에게 보여줄 완성된 응답
            
    사용 예시:
        >>> output = AgentOutput(
        ...     final_output=True,
        ...     progress_description="삼성전자 종목 분석 완료",
        ...     output="삼성전자 분석 결과: 현재 주가는..."
        ... )
    """
    
    final_output: bool = False
    progress_description: str = ""
    output: str = ""
    
    def to_json(self) -> str:
        """객체를 JSON 문자열로 직렬화합니다."""
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'AgentOutput':
        """JSON 문자열에서 객체를 역직렬화합니다."""
        data = json.loads(json_str)
        return cls(**data)

@dataclass
class AgentPlan:
    """
    매니저 에이전트가 수립하는 작업 계획 구조
    
    사용자의 요청을 분석하여 어떤 에이전트들이 어떤 순서로 
    어떤 작업을 수행할지를 체계적으로 계획합니다.
    
    Attributes:
        total_steps (int): 전체 작업 단계 수
        plans (List[Dict[str, Any]]): 단계별 상세 계획
            각 계획은 다음 구조를 가집니다:
            - agent_name: 담당 에이전트 이름
            - description: 구체적인 작업 내용 설명
            - tool_recommendation: 사용 권장 도구 목록
        mode (str): 작업 모드 ('chat' 또는 'agent')
            - chat: 간단한 1-2단계 작업
            - agent: 복잡한 3단계 이상 작업 (데모에서는 제한적 사용)
            
    사용 예시:
        >>> plan = AgentPlan(
        ...     total_steps=2,
        ...     plans=[
        ...         {
        ...             'agent_name': '키키',
        ...             'description': '최신 시장 뉴스 수집 및 분석',
        ...             'tool_recommendation': ['get_latest_news']
        ...         },
        ...         {
        ...             'agent_name': '아거', 
        ...             'description': '삼성전자 종목 심층 분석',
        ...             'tool_recommendation': ['get_stock_price', 'analyze_stock_pattern']
        ...         }
        ...     ],
        ...     mode='chat'
        ... )
    """
    
    total_steps: int = 1
    plans: List[Dict[str, Any]] = field(default_factory=list)
    mode: str = 'chat'
    
    def add_plan_step(self, agent_name: str, description: str, 
                     tool_recommendation: List[str] = None) -> None:
        """
        새로운 작업 단계를 계획에 추가합니다.
        
        Args:
            agent_name (str): 담당할 에이전트 이름
            description (str): 작업 내용 설명
            tool_recommendation (List[str], optional): 권장 도구 목록
            
        사용 예시:
            >>> plan.add_plan_step(
            ...     agent_name='키키',
            ...     description='시장 동향 분석',
            ...     tool_recommendation=['get_latest_news', 'get_market_indicators']
            ... )
        """
        if tool_recommendation is None:
            tool_recommendation = []
            
        plan_step = {
            'agent_name': agent_name,
            'description': description,
            'tool_recommendation': tool_recommendation
        }
        self.plans.append(plan_step)
        self.total_steps = len(self.plans)
    
    def to_json(self) -> str:
        """객체를 JSON 문자열로 직렬화합니다."""
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)
    
    @classmethod  
    def from_json(cls, json_str: str) -> 'AgentPlan':
        """JSON 문자열에서 객체를 역직렬화합니다."""
        data = json.loads(json_str)
        return cls(**data)

@dataclass
class AgentContexts:
    """
    에이전트 작업 컨텍스트를 관리하는 클래스
    
    여러 에이전트가 순차적으로 작업할 때 필요한 상태 정보와
    작업 결과를 체계적으로 관리합니다.
    
    Attributes:
        agent_id_history (List[str]): 호출된 에이전트 순서 기록
        total_step (int): 전체 작업 단계 수 (PLAN에서 설정)
        current_step (int): 현재 진행 중인 단계 번호
        agent_output (List[AgentOutput]): 각 에이전트의 작업 결과 목록
        
    사용 예시:
        >>> agent_ctx = AgentContexts()
        >>> agent_ctx.add_agent_result('키키', AgentOutput(...))
        >>> print(f"현재 {agent_ctx.current_step}/{agent_ctx.total_step} 단계 진행 중")
    """
    
    agent_id_history: List[str] = field(default_factory=list)
    total_step: int = 1
    current_step: int = 0
    agent_output: List[AgentOutput] = field(default_factory=list)
    
    def add_agent_result(self, agent_name: str, output: AgentOutput) -> None:
        """
        에이전트의 작업 결과를 기록합니다.
        
        Args:
            agent_name (str): 작업을 완료한 에이전트 이름
            output (AgentOutput): 에이전트의 작업 결과
            
        사용 예시:
            >>> result = AgentOutput(final_output=False, output="분석 완료...")
            >>> agent_ctx.add_agent_result('키키', result)
        """
        self.agent_id_history.append(agent_name)
        self.agent_output.append(output)
        self.current_step = len(self.agent_id_history)
    
    def get_previous_results(self) -> List[AgentOutput]:
        """
        이전 에이전트들의 작업 결과를 반환합니다.
        
        Returns:
            List[AgentOutput]: 이전 작업 결과 목록
            
        사용 예시:
            >>> previous_results = agent_ctx.get_previous_results()
            >>> for result in previous_results:
            ...     print(f"이전 작업: {result.progress_description}")
        """
        return self.agent_output.copy()
    
    def is_final_step(self) -> bool:
        """
        현재가 마지막 단계인지 확인합니다.
        
        Returns:
            bool: 마지막 단계 여부
        """
        return self.current_step >= self.total_step
    
    def to_json(self) -> str:
        """객체를 JSON 문자열로 직렬화합니다."""
        data = asdict(self)
        # AgentOutput 객체들을 딕셔너리로 변환
        data['agent_output'] = [asdict(output) for output in self.agent_output]
        return json.dumps(data, ensure_ascii=False, indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'AgentContexts':
        """JSON 문자열에서 객체를 역직렬화합니다."""
        data = json.loads(json_str)
        # 딕셔너리들을 AgentOutput 객체로 변환
        if 'agent_output' in data:
            data['agent_output'] = [AgentOutput(**output_dict) 
                                  for output_dict in data['agent_output']]
        return cls(**data)

# 모듈 테스트를 위한 예시 코드
if __name__ == "__main__":
    # 기본 사용법 예시
    print("=== STARGENT 데이터 모델 테스트 ===")
    
    # 1. UserContexts 테스트
    user_ctx = UserContexts(
        user_info={
            'name': '김투자',
            'age': 35,
            'investment_style': '중도적',
            'portfolio': ['삼성전자', 'SK하이닉스'],
            'risk_tolerance': 'medium'
        }
    )
    user_ctx.add_user_message("포트폴리오 분석해줘")
    print(f"✅ 사용자 컨텍스트 생성: {user_ctx.user_info['name']}")
    
    # 2. AgentPlan 테스트
    plan = AgentPlan(mode='chat')
    plan.add_plan_step(
        agent_name='키키',
        description='최신 시장 동향 분석',
        tool_recommendation=['get_latest_news', 'get_market_indicators']
    )
    plan.add_plan_step(
        agent_name='콜리',
        description='포트폴리오 최적화 제안',
        tool_recommendation=[]
    )
    print(f"✅ 작업 계획 생성: {plan.total_steps}단계")
    
    # 3. AgentOutput 테스트
    output = AgentOutput(
        final_output=True,
        progress_description="시장 분석 완료",
        output="현재 시장은 상승 추세를 보이고 있습니다..."
    )
    print(f"✅ 에이전트 출력 생성: {output.progress_description}")
    
    # 4. JSON 직렬화 테스트
    user_json = user_ctx.to_json()
    user_restored = UserContexts.from_json(user_json)
    print(f"✅ JSON 직렬화 테스트: {user_restored.user_info['name']}")
    
    print("\n모든 데이터 모델이 정상적으로 작동합니다! 🎉")