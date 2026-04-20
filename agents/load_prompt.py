from pathlib import Path
import yaml


def load_yaml_prompt(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_system_prompt(path: str) -> str:
    prompt = load_yaml_prompt(path)
    return prompt["messages"][0]["content"]


def get_user_prompt(path: str) -> str:
    prompt = load_yaml_prompt(path)
    return prompt["messages"][1]["content"]
