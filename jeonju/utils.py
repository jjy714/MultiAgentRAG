import base64


def get_base64(bin_file):
    with open(bin_file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()


def get_default_sys_prompt() -> str:
    default_sys_prompt = """### 지시사항
- 친절하게 답변합니다."""
    return default_sys_prompt


def get_rag_sys_prompt(contents: str = "{contents}") -> str:
    rag_sys_prompt = f"""- 참고자료에만 근거해 질문에 답변합니다.
- 답변을 할 때 반드시 참고자료의 파일 명, 페이지, 단락 정보를 함께 알려줍니다.
- 참고자료에 질문에 대한 답변 정보가 있는지 확인 후 친절하게 답합니다.
- 참고자료에 정답이 있는 경우, 참고자료를 활용해 답변을 생성합니다.
- 참고자료에 정답이 없는 경우,"죄송합니다. 주어진 문서에서 사용자 질문에 대한 정보를 확인 할 수 없습니다." 라고만 출력합니다.
### 참고자료
{contents}"""
    return rag_sys_prompt
