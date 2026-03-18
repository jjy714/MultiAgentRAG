from pathlib import Path
import yaml


## Load a YAML prompt file and return its contents as a dictionary
def load_yaml_prompt(path: str) -> dict:
    """
    args   : {
        "path (str)": "file path to the YAML prompt file"
    }
    return : {
        "dict": "parsed YAML content with 'messages' key"
    }
    """
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


## Extract the system-level prompt string from a YAML prompt file
def get_system_prompt(path: str) -> str:
    """
    args   : {
        "path (str)": "file path to the YAML prompt file"
    }
    return : {
        "str": "system message content string"
    }
    """
    prompt = load_yaml_prompt(path)
    return prompt["messages"][0]["content"]


## Extract the user-level prompt string from a YAML prompt file
def get_user_prompt(path: str) -> str:
    """
    args   : {
        "path (str)": "file path to the YAML prompt file"
    }
    return : {
        "str": "user message content string"
    }
    """
    prompt = load_yaml_prompt(path)
    return prompt["messages"][1]["content"]
