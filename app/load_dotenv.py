"""
환경변수 로딩 유틸리티

이 모듈은 프로젝트의 환경변수를 안전하게 로드하는 기능을 제공합니다.
현재 파일 위치를 기준으로 상위 디렉토리의 .env 파일을 찾아 로드합니다.

사용 예시:
    from load_dotenv import load_env
    load_env()  # .env 파일의 환경변수들이 로드됨
"""

import os
from pathlib import Path
from dotenv import load_dotenv as python_dotenv_load

def load_env():
    """
    프로젝트 루트의 .env 파일을 로드합니다.
    
    현재 파일(load_dotenv.py)을 기준으로 상위 디렉토리에서 
    .env 파일을 찾아 환경변수로 로드합니다.
    
    Returns:
        bool: .env 파일 로드 성공 여부
        
    사용 예시:
        >>> load_env()
        True
        >>> import os
        >>> os.getenv('OPENAI_API_KEY')
        'sk-...'
    """
    # 현재 파일의 위치를 기준으로 프로젝트 루트 찾기
    current_file = Path(__file__)
    project_root = current_file.parent.parent  # app/load_dotenv.py -> project_root
    env_path = project_root / '.env'
    
    # .env 파일이 존재하는지 확인
    if env_path.exists():
        result = python_dotenv_load(env_path, override=True)
        if result:
            print(f"✅ 환경변수 로드 완료: {env_path}")
        else:
            print(f"⚠️ 환경변수 로드 실패: {env_path}")
        return result
    else:
        print(f"❌ .env 파일을 찾을 수 없습니다: {env_path}")
        print("💡 .env.example을 참고하여 .env 파일을 생성해주세요.")
        return False


def env_load_test():
    load_env()
    api_key = os.getenv('OPENAI_API_KEY')
    if api_key:
        print(f"✅ OpenAI API 키 확인됨: {api_key[:10]}...")
    else:
        print("❌ OpenAI API 키가 설정되지 않았습니다.")



if __name__ == "__main__":
    # 직접 실행 시 테스트
    env_load_test()