from langchain_core.runnables import RunnableLambda

from .data_model import ThoughtfulResponse
from .parsers import parse_json, parse_sections, parse_thoughts


def chain_json_with_thinking(model, data_schema=None):
    return (
        model
        | parse_sections
        | RunnableLambda(
            lambda sections: ThoughtfulResponse(
                reasoning=parse_thoughts(sections.get("reasoning", "")),
                result=parse_json(sections.get("result", "{}"), data_schema),
            )
        )
    )


def chain_str_with_thinking(model):
    return (
        model
        | parse_sections
        | RunnableLambda(
            lambda sections: ThoughtfulResponse(
                reasoning=parse_thoughts(sections.get("reasoning", "")),
                result=parse_json(sections.get("result", "{}")),
            )
        )
    )


def chain_json_with_validation_thinking(model):
    return (
        model
        | parse_sections
        | RunnableLambda(
            lambda sections: {
                "reasoning": parse_thoughts(sections.get("reasoning", "")),
                "result": (
                    sections.get("result")
                    if sections.get("result") == "ACCEPTED"
                    else parse_json(sections.get("result", "{}"))
                ),
            }
        )
    )
