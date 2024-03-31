import os
import re
import xmltodict

from pydantic.generics import GenericModel
from pydantic import Field
from typing import List, TypeVar, Generic


################ parse LLM output to Pydantic object ################

T = TypeVar("T")


class Result(GenericModel, Generic[T]):
    value: T = Field(alias="result")


def parse_response(response):
    response = response.strip()
    start_tag, end_tag = "<result>", "</result>"
    is_valid = response.startswith(start_tag) and response.endswith(end_tag)
    if not is_valid:
        pattern = f"(?:{start_tag})(.*)(?:{end_tag})"
        p = re.compile(pattern, re.DOTALL)
        m = p.search(response)
        if m is not None:
            response = start_tag + m.group(1) + end_tag
    try:
        resp_dict = xmltodict.parse(response)
    except Exception as e:
        print("response:", response)
        raise e
    result = Result(**resp_dict)
    return result


################ parse LLM output to Pydantic object ################


PROMPT_DIR = "../resources/prompts"


def read_template_from_file(prompt_fn: str,
                            prompt_dir: str = PROMPT_DIR) -> str:
    prompt_fp = os.path.join(prompt_dir, prompt_fn)
    with open(prompt_fp, "r", encoding="utf-8") as f:
        prompt_template_text = f.read()
    return prompt_template_text
