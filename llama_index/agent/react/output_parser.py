"""ReAct output parser."""


import ast
import json
import re
from typing import Tuple

from llama_index.agent.react.types import (
    ActionReasoningStep,
    BaseReasoningStep,
    ResponseReasoningStep,
)
from llama_index.output_parsers.utils import extract_json_str
from llama_index.types import BaseOutputParser



def extract_tool_use(input_text: str, except_thought=False) -> Tuple[str, str, str]:
    pattern = r"\s*Thought:(.*?)Action:(.*?)Action Input:(.*?)(?:\n|$)"
    pattern_except_thought = r"\s*Action:(.*?)Action Input:(.*?)(?:\n|$)"
    match_w_thought = re.search(pattern, input_text, re.DOTALL)
    if not match_w_thought:
        if except_thought:
            match_wo_thought = re.search(pattern_except_thought, input_text, re.DOTALL)
            if not match_wo_thought:
                raise ValueError(f"Could not extract tool use from input text: {input_text}")
        raise ValueError(f"Could not extract tool use from input text: {input_text}")
    if match_w_thought:
        thought = match_w_thought.group(1).strip()
        action = match_w_thought.group(2).strip()
        action_input = match_w_thought.group(3).strip()
    else:
        action = match_wo_thought.group(1).strip()
        action_input = match_wo_thought.group(2).strip()
        thought = ""
    return thought, action, action_input


def extract_final_response(input_text: str, except_thought=False) -> Tuple[str, str]:
    pattern = r"\s*Thought:(.*?)Answer:(.*?)(?:$)"
    pattern_except_thought = r"\s*Answer:(.*?)(?:$)"
    match_w_thought = re.search(pattern, input_text, re.DOTALL)
    

    if not match_w_thought:
        if except_thought:
            match_wo_thought = re.search(pattern_except_thought, input_text, re.DOTALL)
            if not match_wo_thought:
                raise ValueError(f"Could not extract final answer from input text: {input_text}")
        raise ValueError(f"Could not extract final answer from input text: {input_text}")
    if match_w_thought:
        thought = match_w_thought.group(1).strip()
        answer = match_w_thought.group(2).strip()
    else:
        answer = match_wo_thought.group(1).strip()
        thought = ""

    return thought, answer


class ReActOutputParser(BaseOutputParser):
    """ReAct Output parser."""

    def parse(self, output: str, is_streaming: bool = False, except_thought: bool= False) -> BaseReasoningStep:
        """Parse output from ReAct agent.

        We expect the output to be in one of the following formats:
        1. If the agent need to use a tool to answer the question:
            ```
            Thought: <thought>
            Action: <action>
            Action Input: <action_input>
            ```
        2. If the agent can answer the question without any tools:
            ```
            Thought: <thought>
            Answer: <answer>
            ```
        """
        
        if "Answer:" in output:
            thought, answer = extract_final_response(output, except_thought)
            return ResponseReasoningStep(
                thought=thought, response=answer, is_streaming=is_streaming
            )

        if "Action:" in output:
            thought, action, action_input = extract_tool_use(output, except_thought)
            json_str = extract_json_str(action_input)

            # First we try json, if this fails we use ast
            try:
                action_input_dict = json.loads(json_str)
            except json.JSONDecodeError:
                action_input_dict = ast.literal_eval(json_str)

            return ActionReasoningStep(
                thought=thought, action=action, action_input=action_input_dict
            )
        
        if "Thought:" not in output:
            # NOTE: handle the case where the agent directly outputs the answer
            # instead of following the thought-answer format
            return ResponseReasoningStep(
                thought="(Implicit) I can answer without any more tools!",
                response=output,
                is_streaming=is_streaming,
            )
        
        if "Thought:" in output:
            thought_res, _ = extract_final_response(output + " Answer: None\n", except_thought)
            
            return ResponseReasoningStep(
                thought="(Implicit) I can answer without any more tools!",
                response=thought_res, is_streaming=is_streaming
            )
        raise ValueError(f"Could not parse output: {output}")

    def format(self, output: str) -> str:
        """Format a query with structured output formatting instructions."""
        raise NotImplementedError
