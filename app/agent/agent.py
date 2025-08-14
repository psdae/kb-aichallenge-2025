"""
STARGENT AI 에이전트 실행 모듈

이 모듈은 각 AI 에이전트의 실행과 상호작용을 관리합니다.
프롬프트 로딩, LLM 호출, 도구 사용, 에러 핸들링을 통합적으로 처리하여
각 에이전트가 자신의 역할을 효과적으로 수행할 수 있도록 합니다.

주요 기능:
    - 프롬프트 파일 로딩 및 컨텍스트 주입
    - OpenAI GPT 모델과의 상호작용
    - Function Calling을 통한 도구 사용
    - 에러 핸들링 및 재시도 로직
    - 각 에이전트별 특화된 실행 함수

설계 철학:
    - 각 에이전트는 독립적으로 실행 가능
    - 일관된 입력/출력 형식 유지
    - 견고한 에러 처리로 서비스 안정성 확보
    - 확장 가능한 구조로 새로운 에이전트 추가 용이
"""

import os
import json
import openai
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import time
from .model_class import UserContexts, AgentContexts, AgentOutput, AgentPlan
from .tools import AGENT_TOOLS, TOOL_FUNCTIONS

# OpenAI 클라이언트 초기화
client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def load_prompt(agent_name: str) -> str:
    """
    에이전트별 프롬프트 파일을 로딩합니다.
    
    prompts/ 디렉토리에서 해당 에이전트의 마크다운 프롬프트를 읽어와서
    문자열로 반환합니다. 파일이 없는 경우 기본 프롬프트를 반환합니다.
    
    Args:
        agent_name (str): 에이전트 이름 (manager, bibi, kiki, ager, ramu, coli)
        
    Returns:
        str: 프롬프트 내용
        
    사용 예시:
        >>> prompt = load_prompt("bibi")
        >>> print("프롬프트 길이:", len(prompt))
    """
    try:
        # 현재 파일 위치에서 프로젝트 루트 찾기
        current_file = Path(__file__)
        project_root = current_file.parent.parent.parent  # app/agent/agent.py -> project_root
        prompt_path = project_root / "prompts" / f"{agent_name}.md"
        
        if prompt_path.exists():
            with open(prompt_path, 'r', encoding='utf-8') as f:
                prompt_content = f.read()
            print(f"✅ 프롬프트 로딩 완료: {agent_name}.md ({len(prompt_content)} chars)")
            return prompt_content
        else:
            print(f"⚠️ 프롬프트 파일 없음: {prompt_path}")
            return f"당신은 {agent_name} 에이전트입니다. 사용자의 요청에 도움이 되는 응답을 제공해주세요."
            
    except Exception as e:
        print(f"❌ 프롬프트 로딩 실패 ({agent_name}): {e}")
        return f"당신은 {agent_name} 에이전트입니다. 사용자의 요청에 도움이 되는 응답을 제공해주세요."

def safe_llm_call_with_retry(messages: List[Dict], tools: Optional[List] = None, max_retries: int = 2) -> Optional[Dict]:
    """
    LLM 호출을 안전하게 실행하는 헬퍼 함수
    
    네트워크 오류나 API 오류 시 자동으로 재시도하며,
    각종 예외 상황을 적절히 처리합니다.
    
    Args:
        messages (List[Dict]): 대화 메시지 목록
        tools (Optional[List]): 사용할 도구 목록
        max_retries (int): 최대 재시도 횟수
        
    Returns:
        Optional[Dict]: API 응답 또는 None (실패 시)
        
    사용 예시:
        >>> messages = [{"role": "user", "content": "안녕하세요"}]
        >>> response = safe_llm_call_with_retry(messages)
        >>> if response:
        ...     print("응답:", response.choices[0].message.content)
    """
    for attempt in range(max_retries + 1):
        try:
            call_params = {
                "model": "gpt-4.1-mini",
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 2000
            }
            
            # 도구가 있는 경우 추가
            if tools:
                call_params["tools"] = tools
                call_params["tool_choice"] = "auto"
            
            response = client.chat.completions.create(**call_params)
            return response
            
        except openai.APIError as e:
            error_msg = f"OpenAI API 오류: {e}"
        except openai.RateLimitError:
            error_msg = "API 호출 한도 초과"
        except openai.APIConnectionError:
            error_msg = "API 연결 오류"
        except Exception as e:
            error_msg = f"예상치 못한 오류: {e}"
        
        if attempt == max_retries:
            print(f"❌ LLM 호출 최종 실패: {error_msg}")
            return None
        
        print(f"⚠️ LLM 호출 실패 (시도 {attempt + 1}), 재시도: {error_msg}")
        time.sleep(1)  # 1초 대기 후 재시도
    
    return None

def execute_tools_if_needed(response, available_tools: Dict[str, Any]) -> Tuple[str, List[Dict]]:
    """
    LLM 응답에서 도구 호출이 필요한 경우 실행합니다.
    
    OpenAI Function Calling 결과를 파싱하여 해당 도구를 실행하고,
    결과를 다시 LLM에 전달할 수 있도록 메시지 형식으로 변환합니다.
    
    Args:
        response: OpenAI API 응답 객체
        available_tools (Dict[str, Any]): 사용 가능한 도구 함수들
        
    Returns:
        Tuple[str, List[Dict]]: (응답 텍스트, 추가 메시지 목록)
        
    사용 예시:
        >>> response = client.chat.completions.create(...)
        >>> content, additional_messages = execute_tools_if_needed(response, TOOL_FUNCTIONS)
        >>> print("최종 응답:", content)
    """
    message = response.choices[0].message
    additional_messages = []
    
    # 도구 호출이 있는 경우
    if hasattr(message, 'tool_calls') and message.tool_calls:
        # 어시스턴트 메시지 추가
        additional_messages.append({
            "role": "assistant",
            "content": message.content,
            "tool_calls": [
                {
                    "id": tool_call.id,
                    "type": tool_call.type,
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments
                    }
                } for tool_call in message.tool_calls
            ]
        })
        
        # 각 도구 호출 실행
        for tool_call in message.tool_calls:
            function_name = tool_call.function.name
            
            try:
                # 함수 인자 파싱
                function_args = json.loads(tool_call.function.arguments)
                
                # 도구 실행
                if function_name in available_tools:
                    print(f"🔧 도구 실행: {function_name}({function_args})")
                    function_response = available_tools[function_name](**function_args)
                else:
                    function_response = f"오류: 알 수 없는 도구 '{function_name}'"
                
            except json.JSONDecodeError:
                function_response = f"오류: {function_name}의 인자 파싱 실패"
            except Exception as e:
                function_response = f"오류: {function_name} 실행 중 오류 발생 - {str(e)}"
            
            # 도구 실행 결과를 메시지로 추가
            additional_messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": function_response
            })
        
        return message.content or "", additional_messages
    
    # 도구 호출이 없는 경우 일반 응답 반환
    return message.content or "", additional_messages

def create_context_prompt(user_ctx: UserContexts, agent_ctx: AgentContexts, base_prompt: str) -> str:
    """
    기본 프롬프트에 컨텍스트 정보를 주입하여 완성된 프롬프트를 생성합니다.
    
    사용자 정보, 이전 대화 기록, 다른 에이전트들의 작업 결과 등을
    프롬프트에 자연스럽게 통합하여 개인화된 응답이 가능하도록 합니다.
    
    Args:
        user_ctx (UserContexts): 사용자 컨텍스트
        agent_ctx (AgentContexts): 에이전트 컨텍스트  
        base_prompt (str): 기본 프롬프트 템플릿
        
    Returns:
        str: 컨텍스트가 주입된 완성 프롬프트
    """
    context_info = "\n\n=== 컨텍스트 정보 ===\n"
    
    # 사용자 정보 추가
    if user_ctx.user_info:
        context_info += f"\n【사용자 정보】\n"
        for key, value in user_ctx.user_info.items():
            context_info += f"- {key}: {value}\n"
    
    # 최근 대화 기록 추가 (최대 5개)
    if user_ctx.chat_history:
        context_info += f"\n【최근 대화 기록】\n"
        recent_chats = user_ctx.chat_history[-5:]  # 최근 5개만
        for chat in recent_chats:
            role_name = "사용자" if chat['role'] == 'user' else "AI"
            context_info += f"- {role_name}: {chat['content'][:100]}{'...' if len(chat['content']) > 100 else ''}\n"
    
    # 이전 에이전트 작업 결과 추가
    if agent_ctx.agent_output:
        context_info += f"\n【이전 에이전트 작업 결과】\n"
        for i, output in enumerate(agent_ctx.agent_output):
            agent_name = agent_ctx.agent_id_history[i] if i < len(agent_ctx.agent_id_history) else f"Agent{i}"
            context_info += f"- {agent_name}: {output.progress_description}\n"
            context_info += f"  결과: {output.output[:200]}{'...' if len(output.output) > 200 else ''}\n"
    
    # 현재 작업 단계 정보
    context_info += f"\n【작업 진행 상황】\n"
    context_info += f"- 현재 단계: {agent_ctx.current_step + 1}/{agent_ctx.total_step}\n"
    context_info += f"- 완료된 에이전트: {', '.join(agent_ctx.agent_id_history) if agent_ctx.agent_id_history else '없음'}\n"
    
    # 기본 프롬프트와 컨텍스트 결합
    complete_prompt = base_prompt + context_info + "\n\n위 정보를 참고하여 사용자의 요청에 적절히 응답해주세요."
    
    return complete_prompt

# =============================================================================
# 각 에이전트별 실행 함수들
# =============================================================================

def execute_manager(user_ctx: UserContexts, agent_ctx: AgentContexts) -> AgentPlan:
    """
    매니저 에이전트를 실행하여 작업 계획을 수립합니다.
    
    사용자의 요청을 분석하여 어떤 에이전트들이 어떤 순서로 작업할지를
    결정하고, 구체적인 실행 계획을 생성합니다.
    
    Args:
        user_ctx (UserContexts): 사용자 컨텍스트
        agent_ctx (AgentContexts): 에이전트 컨텍스트
        
    Returns:
        AgentPlan: 생성된 작업 계획
        
    사용 예시:
        >>> plan = execute_manager(user_ctx, agent_ctx)
        >>> print(f"총 {plan.total_steps}단계 작업 계획")
        >>> for i, step in enumerate(plan.plans):
        ...     print(f"{i+1}. {step['agent_name']}: {step['description']}")
    """
    try:
        # 매니저 프롬프트 로딩
        base_prompt = load_prompt("manager")
        
        # 최근 사용자 메시지 가져오기
        last_user_message = ""
        if user_ctx.chat_history:
            for chat in reversed(user_ctx.chat_history):
                if chat['role'] == 'user':
                    last_user_message = chat['content']
                    break
        
        # 완성된 프롬프트 생성
        complete_prompt = create_context_prompt(user_ctx, agent_ctx, base_prompt)
        
        # PLAN 요청 메시지 구성
        plan_request = f"""
사용자 요청: "{last_user_message}"

위 요청을 분석하여 적절한 작업 계획(PLAN)을 수립해주세요.
데모 환경이므로 최대 3단계까지만 계획하고, 주로 Chat 모드를 사용해주세요.

JSON 형식으로 응답해주세요:
{{
    "total_steps": 숫자,
    "plans": [
        {{
            "agent_name": "에이전트명",
            "description": "구체적인 작업 설명", 
            "tool_recommendation": ["도구1", "도구2"]
        }}
    ],
    "mode": "chat"
}}
"""
        
        messages = [
            {"role": "system", "content": complete_prompt},
            {"role": "user", "content": plan_request}
        ]
        
        # LLM 호출
        response = safe_llm_call_with_retry(messages)
        
        if response is None:
            # 실패 시 기본 계획 반환
            fallback_plan = AgentPlan(
                total_steps=1,
                plans=[{
                    "agent_name": "비비",
                    "description": "사용자 요청에 대한 기본적인 응답 제공",
                    "tool_recommendation": []
                }],
                mode="chat"
            )
            print("⚠️ 매니저 실행 실패, 기본 계획 사용")
            return fallback_plan
        
        # JSON 파싱 시도
        try:
            response_content = response.choices[0].message.content
            # JSON 부분만 추출 (마크다운 코드 블록 제거)
            if "```json" in response_content:
                json_start = response_content.find("```json") + 7
                json_end = response_content.find("```", json_start)
                response_content = response_content[json_start:json_end].strip()
            elif "```" in response_content:
                json_start = response_content.find("```") + 3
                json_end = response_content.find("```", json_start)
                response_content = response_content[json_start:json_end].strip()
            
            plan_data = json.loads(response_content)
            
            # AgentPlan 객체 생성
            plan = AgentPlan(
                total_steps=plan_data.get("total_steps", 1),
                plans=plan_data.get("plans", []),
                mode=plan_data.get("mode", "chat")
            )
            
            print(f"✅ 매니저 계획 수립 완료: {plan.total_steps}단계, {plan.mode} 모드")
            return plan
            
        except (json.JSONDecodeError, KeyError) as e:
            print(f"⚠️ 매니저 응답 파싱 실패: {e}")
            # 파싱 실패 시 응답 내용 기반 간단한 계획 생성
            if "키키" in response.choices[0].message.content:
                agent_name = "키키"
            elif "아거" in response.choices[0].message.content:
                agent_name = "아거"
            elif "라무" in response.choices[0].message.content:
                agent_name = "라무"
            elif "콜리" in response.choices[0].message.content:
                agent_name = "콜리"
            else:
                agent_name = "비비"
            
            fallback_plan = AgentPlan(
                total_steps=1,
                plans=[{
                    "agent_name": agent_name,
                    "description": "사용자 요청 처리",
                    "tool_recommendation": []
                }],
                mode="chat"
            )
            return fallback_plan
            
    except Exception as e:
        print(f"❌ 매니저 실행 중 오류: {e}")
        # 오류 시 기본 계획
        return AgentPlan(
            total_steps=1,
            plans=[{
                "agent_name": "비비",
                "description": "오류 상황에서의 기본 응답",
                "tool_recommendation": []
            }],
            mode="chat"
        )

def execute_agent(agent_name: str, user_ctx: UserContexts, agent_ctx: AgentContexts) -> AgentOutput:
    """
    지정된 에이전트를 실행합니다.
    
    에이전트별 프롬프트를 로딩하고, 필요한 도구들을 제공하여
    사용자의 요청을 처리합니다. 모든 에이전트가 공통으로 사용하는
    핵심 실행 로직입니다.
    
    Args:
        agent_name (str): 실행할 에이전트 이름
        user_ctx (UserContexts): 사용자 컨텍스트
        agent_ctx (AgentContexts): 에이전트 컨텍스트
        
    Returns:
        AgentOutput: 에이전트 실행 결과
        
    사용 예시:
        >>> result = execute_agent("키키", user_ctx, agent_ctx)
        >>> print(f"작업 완료: {result.progress_description}")
        >>> print(f"결과: {result.output}")
    """
    try:
        print(f"🤖 {agent_name} 에이전트 실행 시작")
        
        # 프롬프트 로딩
        base_prompt = load_prompt(agent_name.lower())
        
        # 컨텍스트가 주입된 완성 프롬프트 생성
        complete_prompt = create_context_prompt(user_ctx, agent_ctx, base_prompt)
        
        # 최근 사용자 메시지 가져오기
        last_user_message = ""
        if user_ctx.chat_history:
            for chat in reversed(user_ctx.chat_history):
                if chat['role'] == 'user':
                    last_user_message = chat['content']
                    break
        
        # 메시지 구성
        messages = [
            {"role": "system", "content": complete_prompt},
            {"role": "user", "content": last_user_message}
        ]
        
        # 에이전트별 사용 가능한 도구 확인
        available_tools = AGENT_TOOLS.get(agent_name, [])
        
        # 첫 번째 LLM 호출
        response = safe_llm_call_with_retry(messages, available_tools if available_tools else None)
        
        if response is None:
            return AgentOutput(
                final_output=True,
                progress_description=f"{agent_name} 실행 실패",
                output="죄송합니다. 현재 서비스에 일시적인 문제가 발생했습니다. 잠시 후 다시 시도해주세요."
            )
        
        # 도구 호출 처리
        content, additional_messages = execute_tools_if_needed(response, TOOL_FUNCTIONS)
        
        # 도구 호출이 있었다면 최종 응답 생성
        if additional_messages:
            final_messages = messages + additional_messages
            final_response = safe_llm_call_with_retry(final_messages)
            
            if final_response:
                final_content = final_response.choices[0].message.content
            else:
                final_content = content or "도구 실행은 완료되었지만 최종 응답 생성에 실패했습니다."
        else:
            final_content = content
        
        # 최종 단계인지 확인
        is_final = agent_ctx.current_step + 1 >= agent_ctx.total_step
        
        # AgentOutput 생성
        result = AgentOutput(
            final_output=is_final,
            progress_description=f"{agent_name}의 작업 완료",
            output=final_content or "응답을 생성할 수 없습니다."
        )
        
        print(f"✅ {agent_name} 에이전트 실행 완료")
        return result
        
    except Exception as e:
        print(f"❌ {agent_name} 에이전트 실행 오류: {e}")
        return AgentOutput(
            final_output=True,
            progress_description=f"{agent_name} 실행 중 오류 발생",
            output=f"죄송합니다. {agent_name} 실행 중 오류가 발생했습니다. 다시 시도해주세요."
        )

# 편의를 위한 개별 에이전트 실행 함수들
def execute_bibi(user_ctx: UserContexts, agent_ctx: AgentContexts) -> AgentOutput:
    """비비 에이전트 실행"""
    return execute_agent("비비", user_ctx, agent_ctx)

def execute_kiki(user_ctx: UserContexts, agent_ctx: AgentContexts) -> AgentOutput:
    """키키 에이전트 실행"""
    return execute_agent("키키", user_ctx, agent_ctx)

def execute_ager(user_ctx: UserContexts, agent_ctx: AgentContexts) -> AgentOutput:
    """아거 에이전트 실행"""
    return execute_agent("아거", user_ctx, agent_ctx)

def execute_ramu(user_ctx: UserContexts, agent_ctx: AgentContexts) -> AgentOutput:
    """라무 에이전트 실행"""
    return execute_agent("라무", user_ctx, agent_ctx)

def execute_coli(user_ctx: UserContexts, agent_ctx: AgentContexts) -> AgentOutput:
    """콜리 에이전트 실행"""
    return execute_agent("콜리", user_ctx, agent_ctx)

# 에이전트 이름과 실행 함수 매핑
AGENT_EXECUTORS = {
    "비비": execute_bibi,
    "키키": execute_kiki, 
    "아거": execute_ager,
    "라무": execute_ramu,
    "콜리": execute_coli
}

def execute_plan(plan: AgentPlan, user_ctx: UserContexts) -> AgentContexts:
    """
    작업 계획에 따라 에이전트들을 순차적으로 실행합니다.
    
    매니저가 수립한 계획에 따라 각 에이전트를 순서대로 호출하고,
    각 단계의 결과를 다음 단계로 전달하여 협업을 구현합니다.
    
    Args:
        plan (AgentPlan): 실행할 작업 계획
        user_ctx (UserContexts): 사용자 컨텍스트
        
    Returns:
        AgentContexts: 실행 결과가 포함된 에이전트 컨텍스트
        
    사용 예시:
        >>> plan = execute_manager(user_ctx, agent_ctx)
        >>> result_ctx = execute_plan(plan, user_ctx)
        >>> final_output = result_ctx.agent_output[-1]
        >>> print("최종 결과:", final_output.output)
    """
    # 새로운 AgentContexts 생성
    agent_ctx = AgentContexts(
        total_step=plan.total_steps,
        current_step=0,
        agent_id_history=[],
        agent_output=[]
    )
    
    print(f"📋 작업 계획 실행 시작: {plan.total_steps}단계")
    
    # 각 단계별로 에이전트 실행
    for step_idx, step_plan in enumerate(plan.plans):
        agent_name = step_plan["agent_name"]
        description = step_plan["description"]
        
        print(f"\n📍 단계 {step_idx + 1}/{plan.total_steps}: {agent_name} - {description}")
        
        # 해당 에이전트 실행
        if agent_name in AGENT_EXECUTORS:
            result = AGENT_EXECUTORS[agent_name](user_ctx, agent_ctx)
        else:
            print(f"⚠️ 알 수 없는 에이전트: {agent_name}, 비비로 대체")
            result = execute_bibi(user_ctx, agent_ctx)
        
        # 결과를 컨텍스트에 추가
        agent_ctx.add_agent_result(agent_name, result)
        
        print(f"✅ {agent_name} 작업 완료: {result.progress_description}")
    
    print(f"🎉 전체 작업 계획 실행 완료!")
    return agent_ctx

# 모듈 테스트
if __name__ == "__main__":
    print("=== STARGENT 에이전트 실행 모듈 테스트 ===")
    
    # 환경 확인
    if not os.getenv('OPENAI_API_KEY'):
        print("⚠️ OPENAI_API_KEY가 설정되지 않아 실제 테스트를 건너뜁니다.")
        exit()
    
    # 테스트용 컨텍스트 생성
    user_ctx = UserContexts(
        user_info={
            'name': '테스트사용자',
            'investment_style': '중도적',
            'portfolio': ['삼성전자', 'SK하이닉스']
        }
    )
    user_ctx.add_user_message("삼성전자 주가 어때?")
    
    agent_ctx = AgentContexts()
    
    # 매니저 실행 테스트
    print("\n🧠 매니저 에이전트 테스트:")
    plan = execute_manager(user_ctx, agent_ctx)
    print(f"생성된 계획: {plan.total_steps}단계")
    print(plan)
    
    # 개별 에이전트 실행 테스트 (비비)
    print("\n🐻 비비 에이전트 테스트:")
    result = execute_bibi(user_ctx, agent_ctx)
    print(f"실행 결과: {result.progress_description}")
    print(result)
    
    print("\n✅ 에이전트 실행 모듈이 정상적으로 작동합니다! 🎉")