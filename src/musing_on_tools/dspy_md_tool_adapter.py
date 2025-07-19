#%% [markdown]
#
# # Making a custom dspy adapter for ToolCalling using # | and md code blocks
#
#



#%%
import re
import textwrap
from typing import Any, Dict, Optional, Type, NamedTuple

from litellm import ContextWindowExceededError
from pydantic.fields import FieldInfo

from dspy.adapters.base import Adapter
from dspy.adapters.utils import (
    format_field_value,
    get_annotation_name,
    parse_value,
)
from dspy.clients.lm import LM
from dspy.signatures.signature import Signature
from dspy.utils.callback import BaseCallback
from dspy.utils.exceptions import AdapterParseError


# Sentinel that marks the end of a field block when there are **multiple** outputs
END_SENTINEL = "<!-- end -->"

class MarkdownAdapter(Adapter):
    """Markdown adapter for structured output using headings and sentinels.

    Output format:
    - Single field: Return value directly
    - Multiple fields: Use markdown headings (# field_name) followed by content,
      then <!-- end --> on its own line to mark the end of each field
    """

    # ──────────────────────────────────────────────────────────────────────────────
    # Boilerplate & fall‑back to JSONAdapter on context overflow
    # ──────────────────────────────────────────────────────────────────────────────

    def __init__(self, callbacks: Optional[list[BaseCallback]] = None):
        super().__init__(callbacks)

    def __call__(
        self,
        lm: LM,
        lm_kwargs: dict[str, Any],
        signature: Type[Signature],
        demos: list[dict[str, Any]],
        inputs: dict[str, Any],
    ) -> list[dict[str, Any]]:
        try:
            return super().__call__(lm, lm_kwargs, signature, demos, inputs)
        except (ContextWindowExceededError, AdapterParseError):
            from dspy.adapters.json_adapter import JSONAdapter
            return JSONAdapter()(lm, lm_kwargs, signature, demos, inputs)

    # ──────────────────────────────────────────────────────────────────────────────
    # Prompt helpers
    # ──────────────────────────────────────────────────────────────────────────────

    def format_field_description(self, signature: Type[Signature]) -> str:
        inputs = ", ".join(signature.input_fields)
        outputs = ", ".join(signature.output_fields)
        return f"Inputs: {inputs}\nOutputs: {outputs}"

    def format_user_message_content(
        self,
        signature: Type[Signature],
        inputs: Optional[dict[str, Any]] = None,
        prefix: str = "",
        suffix: str = "",
        main_request: bool = False,
        *_, **__,
    ) -> str:
        """Compose the user message for the LM.
        The *trajectory‑only* call path may supply `inputs=None`, so we guard for that.
        """
        inputs = inputs or {}
        parts: list[str] = [prefix] if prefix else []

        for key, val in inputs.items():
            if key not in signature.input_fields:
                continue  # ignore things like trajectory keys
            parts.append(f"### {key}\n{format_field_value(signature.input_fields[key], val)}")

        if main_request:
            parts.append(self.user_message_output_requirements(signature))
        if suffix:
            parts.append(suffix)
        return "\n\n".join(parts).strip()

    def user_message_output_requirements(self, signature: Type[Signature]) -> str:
        if len(signature.output_fields) == 1:
            field = next(iter(signature.output_fields))
            return (
                f"Respond with **only** the value for `{field}` — no heading needed."
            )
        else:
            field_lines = "\n".join(
                f"* `# {name}` (or any heading level) … {END_SENTINEL}"  # guidance list
                for name in signature.output_fields
            )
            return (
                "Format each output block as:\n"
                "    # <field_name>\n"
                "    <value>\n"
                f"    {END_SENTINEL}\n\n"
                "The heading text must exactly match the field name; use *any* heading level."
            )

    def format_assistant_message_content(
        self,
        signature: Type[Signature],
        outputs: dict[str, Any],
        missing_field_message=None,
    ) -> str:
        """Reference implementation for demos / finetune data."""
        if len(outputs) == 1:
            return next(iter(outputs.values()))
        blocks: list[str] = []
        for name, info in signature.output_fields.items():
            val = outputs.get(name, missing_field_message)
            blocks.append(
                f"### {name}\n{format_field_value(info, val)}\n{END_SENTINEL}"
            )
        return "\n\n".join(blocks)

    def format_field_structure(self, signature: Type[Signature]) -> str:
        """Describe the exact input/output wire-format for the language model.

        This string becomes part of the system prompt and explains *how* the model
        should lay out the fields.  For our Markdown-style adapter we keep the
        description short and pragmatic – large models do not need excessive
        verbosity.
        """
        if len(signature.output_fields) == 1:
            field = next(iter(signature.output_fields))
            return (
                "Reply with the **value only** for the output field `"
                f"{field}` – no markdown heading, no surrounding commentary."
            )
        else:
            sentinel = END_SENTINEL
            return (
                "For *each* output field, reply using a markdown block that looks like:\n"
                "    # <field_name>\n"
                "    <value>\n"
                f"    {sentinel}\n\n"
                "You may pick any heading depth (e.g. `#`, `##`, ...), but the heading text\n"
                "must match the field name *exactly*.  Finish **every** block with the sentinel\n"
                f"`{sentinel}` on its own line."
            )

    def format_task_description(self, signature: Type[Signature]) -> str:
        """Return the high-level instructions for the task (verbatim from the signature)."""
        return textwrap.dedent(signature.instructions or "").strip()

    # ──────────────────────────────────────────────────────────────────────────────
    # Parsing logic
    # ──────────────────────────────────────────────────────────────────────────────

    def _build_regex(self, signature: Type[Signature]):
        """Compile a single regex that grabs *heading, body* pairs up to the sentinel."""
        names_alt = "|".join(map(re.escape, signature.output_fields))
        return re.compile(
            rf"^#+\s*({names_alt})\s*$"       # heading (any level)
            rf"([\s\S]*?)"                    # body (non‑greedy)
            rf"^\s*{re.escape(END_SENTINEL)}\s*$",  # sentinel line
            re.MULTILINE,
        )

    def parse(self, signature: Type[Signature], completion: str) -> Dict[str, Any]:
        outputs = signature.output_fields
        # Fast path: one output → whole text.
        if len(outputs) == 1:
            single_key = next(iter(outputs))
            return {single_key: parse_value(completion.strip(), outputs[single_key].annotation)}

        regex = self._build_regex(signature)
        parsed: Dict[str, Any] = {}

        for match in regex.finditer(completion):
            field_name, body = match.group(1), match.group(2)
            if field_name in outputs and field_name not in parsed:
                parsed[field_name] = parse_value(body.strip(), outputs[field_name].annotation)

        missing = set(outputs) - set(parsed)
        if missing:
            raise AdapterParseError(
                adapter_name="MarkdownAdapter",
                signature=signature,
                lm_response=completion,
                parsed_result=parsed,
            )
        return parsed

    # ──────────────────────────────────────────────────────────────────────────────
    # Finetune data helper (same pattern as ChatAdapter)
    # ──────────────────────────────────────────────────────────────────────────────

    def format_finetune_data(
        self,
        signature: Type[Signature],
        demos: list[dict[str, Any]],
        inputs: dict[str, Any],
        outputs: dict[str, Any],
    ) -> dict[str, list[Any]]:
        messages = self.format(signature, demos, inputs)
        assistant_content = self.format_assistant_message_content(signature, outputs)
        messages.append({"role": "assistant", "content": assistant_content})
        return {"messages": messages}


#%%



# %%
import logging
from typing import Any, Callable, Literal

from litellm import ContextWindowExceededError

import dspy
from dspy.adapters.types.tool import Tool
from dspy.primitives.program import Module
from dspy.signatures.signature import ensure_signature

logger = logging.getLogger(__name__)


class ReAct_md(Module):
    def __init__(
        self,
        signature,
        tools: Optional[list[Callable]] = None,
        max_iters: int = 5,
        *,
        run_mode: str = "strict",  # "strict" | "guided" | "unrestricted"
        guided_allowlist: Optional[set[str]] = None,
    ):
        """
        `tools` is either a list of functions, callable classes, or `dspy.Tool` instances.
        """
        super().__init__()
        self.signature = signature = ensure_signature(signature)
        self.max_iters = max_iters
        self.run_mode = run_mode
        if run_mode not in {"strict", "guided", "unrestricted"}:
            raise ValueError("run_mode must be 'strict', 'guided', or 'unrestricted'")
        self.guided_allowlist = set(guided_allowlist or [])

        tools = tools or []
        tools = [t if isinstance(t, Tool) else Tool(t) for t in tools]
        tools = {tool.name: tool for tool in tools}

        inputs = ", ".join([f"`{k}`" for k in signature.input_fields.keys()])
        outputs = ", ".join([f"`{k}`" for k in signature.output_fields.keys()])
        instr = [f"{signature.instructions}\n"] if signature.instructions else []

        instr.extend(
            [
                f"You are an Agent. In each episode, you will be given the fields {inputs} as input. And you can see your past trajectory so far.",
                f"Your goal is to use one or more of the supplied tools to collect any necessary information for producing {outputs}.\n",
                "To do this, you will interleave next_thought, next_tool_name, and next_tool_args in each turn, and also when finishing the task.",
                "After each tool call, you receive a resulting observation, which gets appended to your trajectory.\n",
                "When writing next_thought, you may reason about the current situation and plan for future steps.",
                "When selecting the next_tool_name and its next_tool_args, the tool must be one of:\n",
            ]
        )

        tools["finish"] = Tool(
            func=lambda: "Completed.",
            name="finish",
            desc=f"Marks the task as complete. That is, signals that all information for producing the outputs, i.e. {outputs}, are now available to be extracted.",
            args={},
        )

        for idx, tool in enumerate(tools.values()):
            instr.append(f"({idx + 1}) {tool}")

        # ------------------------------------------------------------------
        # Mode-specific guidance
        # ------------------------------------------------------------------
        common_block_example = (
            "```python\n# | run\nget_current_time()\n```"
        )

        if self.run_mode == "strict":
            mode_expl = (
                "You are in *STRICT* execution mode: you may only call the numbered tools listed above. "
                "Attempting to call anything else will raise an error."
            )
        elif self.run_mode == "guided":
            allow_txt = ", ".join(sorted(self.guided_allowlist)) or "<none>"
            mode_expl = (
                "You are in *GUIDED* execution mode: in addition to the numbered tools, you may call any function whose name "
                f"is in the guided allow-list: {allow_txt}. Attempts outside this list will error."
            )
        else:  # unrestricted
            mode_expl = (
                "You are in *UNRESTRICTED* execution mode: you may place arbitrary valid Python code after `# | run`. "
                "The expression will be evaluated and its result returned. Use with care."
            )

        instr.append(
            mode_expl
            + "\n\nWhen you need to execute something, **output only a markdown python code block** that starts with `# | run`, "
            "following this example:\n\n"
            f"{common_block_example}\n\n"
            "Put this code block in the `assistant_response` field, and **do NOT** include any other text in that field.\n"
            "Your `next_thought` is for your private reasoning and must NOT include run-cells.\n"
            "When all necessary executions are complete and you are ready to answer the user, return the final answer "
            "in plain language in `assistant_response` with no code block."
        )

        react_signature = (
            dspy.Signature({**signature.input_fields}, "\n".join(instr))
            .append("trajectory", dspy.InputField(), type_=str)
            .append("next_thought", dspy.OutputField(), type_=str)
            .append("assistant_response", dspy.OutputField(), type_=str)
        )

        fallback_signature = dspy.Signature(
            {**signature.input_fields, **signature.output_fields},
            signature.instructions,
        ).append("trajectory", dspy.InputField(), type_=str)

        self.tools = tools
        self.react = dspy.Predict(react_signature)
        self.extract = dspy.ChainOfThought(fallback_signature)

    def _format_trajectory(self, trajectory: dict[str, Any]):
        adapter = dspy.settings.adapter or dspy.ChatAdapter()
        trajectory_signature = dspy.Signature(f"{', '.join(trajectory.keys())} -> x")
        return adapter.format_user_message_content(trajectory_signature, trajectory)

    def _process_assistant_response(self, response: str, trajectory: dict[str, Any], extra_scan: str | None = None) -> bool:
        """Execute any run-cells found in *response* or *extra_scan*.

        If a finish() tool call is executed or no run-cells are present, returns True to indicate the episode should end.
        """
        finished = False
        scan_sources = [response]
        if extra_scan:
            scan_sources.append(extra_scan)
        run_blocks: list[str] = []
        for src in scan_sources:
            if src:
                run_blocks.extend(_find_run_blocks(src))

        if not run_blocks:
            # No tool call – assume assistant is providing final answer.
            return True

        for code in run_blocks:
            if self.run_mode == "unrestricted":
                result_block = self._execute_unrestricted_block(code)
            else:
                try:
                    call = _parse_tool_call(code)
                except Exception as err:
                    if self.run_mode == "unrestricted":
                        # Multi-line or non-call cell: treat as unrestricted code block
                        result_block = self._execute_unrestricted_block(code)
                    else:
                        result_block = _RESULT_TEMPLATE.format(tool_name="parse_error", result=str(err))
                else:
                    if call.name == "finish":
                        finished = True
                    result_block = self._execute_tool_call_with_mode(code, call)
            trajectory_key = f"observation_{len(trajectory)}"
            trajectory[trajectory_key] = result_block
        return finished

    def forward(self, **input_args):
        trajectory: dict[str, Any] = {}
        max_iters = input_args.pop("max_iters", self.max_iters)
        for idx in range(max_iters):
            pred = self._call_with_potential_trajectory_truncation(self.react, trajectory, **input_args)
            trajectory[f"thought_{idx}"] = pred.next_thought
            trajectory[f"assistant_response_{idx}"] = pred.assistant_response
            done = self._process_assistant_response(pred.assistant_response, trajectory, extra_scan=pred.next_thought)
            if done:
                break

        extract = self._call_with_potential_trajectory_truncation(self.extract, trajectory, **input_args)
        return dspy.Prediction(trajectory=trajectory, **extract)

    async def aforward(self, **input_args):
        trajectory: dict[str, Any] = {}
        max_iters = input_args.pop("max_iters", self.max_iters)
        for idx in range(max_iters):
            pred = await self._async_call_with_potential_trajectory_truncation(self.react, trajectory, **input_args)
            trajectory[f"thought_{idx}"] = pred.next_thought
            trajectory[f"assistant_response_{idx}"] = pred.assistant_response
            done = self._process_assistant_response(pred.assistant_response, trajectory, extra_scan=pred.next_thought)
            if done:
                break

        extract = await self._async_call_with_potential_trajectory_truncation(self.extract, trajectory, **input_args)
        return dspy.Prediction(trajectory=trajectory, **extract)

    def _call_with_potential_trajectory_truncation(self, module, trajectory, **input_args):
        for _ in range(3):
            try:
                return module(
                    **input_args,
                    trajectory=self._format_trajectory(trajectory),
                )
            except ContextWindowExceededError:
                logger.warning("Trajectory exceeded the context window, truncating the oldest tool call information.")
                trajectory = self.truncate_trajectory(trajectory)

    async def _async_call_with_potential_trajectory_truncation(self, module, trajectory, **input_args):
        for _ in range(3):
            try:
                return await module.acall(
                    **input_args,
                    trajectory=self._format_trajectory(trajectory),
                )
            except ContextWindowExceededError:
                logger.warning("Trajectory exceeded the context window, truncating the oldest tool call information.")
                trajectory = self.truncate_trajectory(trajectory)

    def truncate_trajectory(self, trajectory):
        """Truncates the trajectory so that it fits in the context window.

        Users can override this method to implement their own truncation logic.
        """
        keys = list(trajectory.keys())
        if len(keys) < 4:
            # Every tool call has 4 keys: thought, tool_name, tool_args, and observation.
            raise ValueError(
                "The trajectory is too long so your prompt exceeded the context window, but the trajectory cannot be "
                "truncated because it only has one tool call."
            )

        for key in keys[:4]:
            trajectory.pop(key)

        return trajectory

    # ------------------------------------------------------------------
    # Utility: execute according to run_mode
    # ------------------------------------------------------------------

    def _execute_tool_call_with_mode(self, code: str, call: "_ParsedToolCall") -> str:
        """Execute the parsed call in accordance with self.run_mode."""

        # finish is always allowed and uses strict path
        if call.name == "finish":
            return _execute_tool_call(call, self.tools)

        if self.run_mode == "strict":
            return _execute_tool_call(call, self.tools)

        if self.run_mode == "guided":
            if call.name in self.tools:
                return _execute_tool_call(call, self.tools)
            if call.name not in self.guided_allowlist:
                return _RESULT_TEMPLATE.format(
                    tool_name=call.name,
                    result=f"Error: '{call.name}' not in guided allowlist.",
                )

        # At this stage we allow execution of arbitrary python expression (code)
        safe_globals: dict[str, Any] = {n: t.func for n, t in self.tools.items()}
        safe_builtins = {
            "__import__": __import__,
            "print": print,
            "len": len,
            "range": range,
            "min": min,
            "max": max,
            "sum": sum,
        }
        safe_globals["__builtins__"] = safe_builtins

        local_ns: dict[str, Any] = {}

        import io, contextlib

        stdout_buffer = io.StringIO()

        try:
            with contextlib.redirect_stdout(stdout_buffer):
                compiled = compile(code, "<run_cell>", "exec")
                exec(compiled, safe_globals, local_ns)

            printed_output = stdout_buffer.getvalue().strip()

            # Attempt to get the value of the last expression if it exists
            result_value = None
            try:
                tree = ast.parse(code, mode="exec")
                if tree.body and isinstance(tree.body[-1], ast.Expr):
                    expr = tree.body[-1].value
                    expr_code = compile(ast.Expression(expr), "<run_cell_expr>", "eval")
                    result_value = eval(expr_code, safe_globals, local_ns)
            except Exception:
                pass

            if result_value is None:
                res = printed_output if printed_output else None
            else:
                res = result_value

            if isinstance(res, str):
                res_str = res
            else:
                try:
                    res_str = json.dumps(res, ensure_ascii=False)
                except TypeError:
                    res_str = str(res)
        except Exception as err:
            res_str = f"Execution error: {err}"
        return _RESULT_TEMPLATE.format(tool_name=call.name, result=res_str)

    # ------------------------------------------------------------------
    # Unrestricted execution helper (multiple statements supported)
    # ------------------------------------------------------------------

    def _execute_unrestricted_block(self, code: str) -> str:
        """Executes arbitrary Python *code* and returns a result block.

        The last expression's value (if any) is captured as the result. All named tools are available in the exec
        environment as their underlying functions. Built-ins are removed for safety.
        """

        safe_globals: dict[str, Any] = {n: t.func for n, t in self.tools.items()}
        safe_builtins = {
            "__import__": __import__,
            "print": print,
            "len": len,
            "range": range,
            "min": min,
            "max": max,
            "sum": sum,
        }
        safe_globals["__builtins__"] = safe_builtins

        local_ns: dict[str, Any] = {}

        import io, contextlib

        stdout_buffer = io.StringIO()

        try:
            with contextlib.redirect_stdout(stdout_buffer):
                compiled = compile(code, "<run_cell>", "exec")
                exec(compiled, safe_globals, local_ns)

            printed_output = stdout_buffer.getvalue().strip()

            # Attempt to get the value of the last expression if it exists
            result_value = None
            try:
                tree = ast.parse(code, mode="exec")
                if tree.body and isinstance(tree.body[-1], ast.Expr):
                    expr = tree.body[-1].value
                    expr_code = compile(ast.Expression(expr), "<run_cell_expr>", "eval")
                    result_value = eval(expr_code, safe_globals, local_ns)
            except Exception:
                pass

            if result_value is None:
                res = printed_output if printed_output else None
            else:
                res = result_value

            if isinstance(res, str):
                res_str = res
            else:
                try:
                    res_str = json.dumps(res, ensure_ascii=False)
                except TypeError:
                    res_str = str(res)
        except Exception as err:
            res_str = f"Execution error: {err}"
        return _RESULT_TEMPLATE.format(tool_name="code_block", result=res_str)


def _fmt_exc(err: BaseException, *, limit: int = 5) -> str:
    """
    Return a one-string traceback summary.
    * `limit` - how many stack frames to keep (from the innermost outwards).
    """

    import traceback

    return "\n" + "".join(traceback.format_exception(type(err), err, err.__traceback__, limit=limit)).strip()


#%% NEW IMPORTS FOR RUN-CELL PARSING
import ast
import json

#%%
# -----------------------------------------------------------------------------
# Utilities for the new "# | run" code-block protocol
# -----------------------------------------------------------------------------
RUN_BLOCK_PATTERN = re.compile(
    r"```python\s*#\s*\|\s*run\s*\n(?P<code>[\s\S]+?)```",
    re.IGNORECASE,
)


def _find_run_blocks(text: str) -> list[str]:
    """Return raw code for every run-cell.

    Primarily we look for the canonical fenced code block of the form::

        ```python
        # | run
        ...
        ```

    However, some model responses omit the surrounding markdown fence and only
    emit lines starting with ``# | run`` followed by the code.  To make the
    system more robust we also capture those *un-fenced* variants.
    """

    # 1. Standard fenced blocks
    run_blocks: list[str] = [m.group("code").strip() for m in RUN_BLOCK_PATTERN.finditer(text)]

    # 2. Fallback – un-fenced variant beginning with "# | run"
    if not run_blocks:
        lines = text.splitlines()
        i = 0
        while i < len(lines):
            if re.match(r"\s*#\s*\|\s*run", lines[i], flags=re.IGNORECASE):
                # Collect all subsequent lines until another directive-like
                # header (e.g. "# | result") or end-of-text.
                j = i + 1
                code_lines: list[str] = []
                while j < len(lines) and not re.match(r"\s*#\s*\|", lines[j]):
                    code_lines.append(lines[j])
                    j += 1
                if code_lines:
                    run_blocks.append("\n".join(code_lines).strip())
                i = j  # continue scanning after the collected block
            else:
                i += 1

    return run_blocks


class _ParsedToolCall(NamedTuple):
    name: str
    kwargs: dict[str, Any]


def _parse_tool_call(code: str) -> _ParsedToolCall:
    """Parse the first line of *code* into a tool call.

    Restrictions:
    • Exactly one top-level Call expression.
    • Only keyword or no arguments. Positional arguments raise.
    • Argument values must be literal python expressions (ast.literal_eval).
    """

    try:
        module = ast.parse(code, mode="exec")
    except SyntaxError as err:
        raise ValueError(f"Invalid python code in run-cell: {err.msg}") from err

    if len(module.body) != 1 or not isinstance(module.body[0], ast.Expr):
        raise ValueError("Run-cell must contain exactly one expression statement.")

    call_node = module.body[0].value
    if not isinstance(call_node, ast.Call) or not isinstance(call_node.func, ast.Name):
        raise ValueError("Run-cell must be a direct function call like foo(a=1).")

    if call_node.args:
        raise ValueError("Only keyword arguments are supported in run-cells.")

    kwargs = {}
    for kw in call_node.keywords:
        if kw.arg is None:
            raise ValueError("**kwargs not supported in run-cells.")
        kwargs[kw.arg] = ast.literal_eval(kw.value)

    return _ParsedToolCall(call_node.func.id, kwargs)


_RESULT_TEMPLATE = """```python\n# | result\n{tool_name} -> {result}\n```"""


def _execute_tool_call(call: _ParsedToolCall, tools: dict[str, "Tool"]) -> str:
    """Execute *call* against *tools* and return a formatted result block."""
    if call.name not in tools:
        return _RESULT_TEMPLATE.format(tool_name=call.name, result=f"Error: Unknown tool '{call.name}'.")
    try:
        res = tools[call.name](**call.kwargs)
        res_str = json.dumps(res, ensure_ascii=False) if not isinstance(res, str) else res
    except Exception as err:
        res_str = f"Execution error: {err}"
    return _RESULT_TEMPLATE.format(tool_name=call.name, result=res_str)

# %%



# %%

# # %%
from typing import Any


def evaluate_code(code: str) -> Any:
    """Evaluate a string of Python code and return the result.
    
    Args:
        code: A string containing Python code to evaluate
        
    Returns:
        The result of evaluating the code
        
    Raises:
        ValueError: If the code is invalid or cannot be evaluated
    """
    import ast
    import io
    import contextlib
    
    # Create a safe environment with basic builtins
    safe_builtins = {
        "__import__": __import__,
        "print": print,
        "len": len,
        "range": range,
        "min": min,
        "max": max,
        "sum": sum,
        "abs": abs,
        "round": round,
        "pow": pow,
        "int": int,
        "float": float,
        "str": str,
        "bool": bool,
        "list": list,
        "dict": dict,
        "tuple": tuple,
        "set": set,
    }
    
    safe_globals = {"__builtins__": safe_builtins}
    local_ns = {}
    
    # Capture stdout
    stdout_buffer = io.StringIO()
    
    try:
        with contextlib.redirect_stdout(stdout_buffer):
            # Try to compile and execute as a statement first
            compiled = compile(code, "<evaluate_code>", "exec")
            exec(compiled, safe_globals, local_ns)
        
        printed_output = stdout_buffer.getvalue().strip()
        
        # Try to get the value of the last expression if it exists
        result_value = None
        try:
            tree = ast.parse(code, mode="exec")
            if tree.body and isinstance(tree.body[-1], ast.Expr):
                expr = tree.body[-1].value
                expr_code = compile(ast.Expression(expr), "<evaluate_code_expr>", "eval")
                result_value = eval(expr_code, safe_globals, local_ns)
        except Exception:
            pass
        
        # Return the expression value if available, otherwise printed output
        if result_value is not None:
            return result_value
        elif printed_output:
            return printed_output
        else:
            return None
            
    except Exception as err:
        raise ValueError(f"Code evaluation error: {err}") from err






import dspy
lm = dspy.LM(model="groq/llama3-8b-8192")
dspy.configure(lm = lm, adapter=MarkdownAdapter())

react = dspy.ReAct("question -> answer", tools=[evaluate_code])

r_tooljson = react(question="What time is it?")
print(r_tooljson.answer)
r_tooljson.trajectory
#%%
import dspy
lm = dspy.LM(model="groq/llama3-8b-8192")
dspy.configure(lm = lm, adapter=MarkdownAdapter())

react_md = ReAct_md("question -> answer", run_mode="unrestricted")

r_md = react_md(question="What time is it?")
print(r_md.answer)
r_md.trajectory


#%%

import dspy
lm = dspy.LM(model="groq/llama3-8b-8192")
dspy.configure(lm = lm, adapter=MarkdownAdapter())


react = ReAct_md("question -> answer", run_mode="unrestricted")

r_md = react(question="Calculate the compound interest for 1000000 at 2% for 10 years")
r_md.trajectory


#%%
import dspy
import os
os.environ['OPENAI_BASE_URL'] = "http://192.168.147.194:1234/v1"
lm = dspy.LM(model="openai/qwen/qwen3-1.7b")
dspy.configure(lm = lm, adapter=MarkdownAdapter())
#%%
react = ReAct_md("question -> answer", run_mode="unrestricted")
r_md = react(question="/no_think Calculate the compound interest for 1000000 at 2% for 10 years")
r_md.trajectory
# %%
lm(" hello")
# %%




import dspy
lm = dspy.LM(model="groq/llama3-8b-8192")
dspy.configure(lm = lm, adapter=MarkdownAdapter())
react = ReAct_md("question -> answer", run_mode="unrestricted")
r_md = react(question="What is the biggest file in the current directory?")

# Prediction(
#     trajectory={
#             "thought_0": "I need to get the list of files in the current directory and then find the largest one.",
#             "assistant_response_0": "# | run\nimport os\nfiles = os.listdir()\nmax_file_size = 0\nmax_file = None\nfor file in files:\n    file_path = os.path.join(os.getcwd(), file)\n    if os.path.isfile(file_path):\n        file_size = os.path.getsize(file_path)\n        if file_size > max_file_size:\n            max_file_size = file_size\n            max_file = file",
#             "observation_0": "```python\n# | result\ncode_block -> null\n```",
#             "thought_1": "I need to return the name of the largest file.",
#             "assistant_response_1": "# | run\nprint(max_file)",
#             "observation_1": "```python\n# | result\ncode_block -> Execution error: name 'max_file' is not defined\n```",
#             "thought_2": "I need to define max_file before using it.",
#             "assistant_response_2": "# | run\nimport os\nfiles = os.listdir()\nmax_file_size = 0\nmax_file = None\nfor file in files:\n    file_path = os.path.join(os.getcwd(), file)\n    if os.path.isfile(file_path):\n        file_size = os.path.getsize(file_path)\n        if file_size > max_file_size:\n            max_file_size = file_size\n            max_file = file\nprint(max_file)",
#             "observation_2": "```python\n# | result\ncode_block -> mcp_trends_infographic.png\n```",
#             "thought_3": "I have found the largest file in the current directory, now I need to return its name.",
#             "assistant_response_3": ""
#         },
#     reasoning='To find the biggest file in the current directory, we need to list all files in the directory, then iterate through the list to find the file with the largest size. We use the `os` module to interact with the file system. We define `max_file` as `None` initially and update it whenever we find a file with a larger size. Finally, we print the name of the largest file.',
#     answer='mcp_trends_infographic.png'
# )

#%%
import dspy
lm = dspy.LM(model="groq/llama3-8b-8192")
dspy.configure(lm = lm, adapter=MarkdownAdapter())
react = ReAct_md("question -> answer", run_mode="unrestricted")
r_md = react(question="What files are in the current directory?")
print(r_md.answer)
#The files in the current directory are: 
#['multiTimeline (3).csv', 'multiTimeline (2).csv', 'plotting_mcp_events.R', 'mcp_trends_infographic.svg', 'multiTimeline.csv', 'multiTimeline (1).csv', 'dspy_md_tool_adapter.py', 'mcp_trends_infographic.png', 'Screenshot from 2025-06-16 14-47-42.png', 'Screenshot from 2025-06-16 13-59-26.png', 'tool_use_trends.svg', 'chapter1.py', 'tool_use_trends.png']
# %%

#%%
import dspy
lm = dspy.LM(model="groq/meta-llama/llama-4-maverick-17b-128e-instruct")
dspy.configure(lm = lm, adapter=MarkdownAdapter())
react = ReAct_md("question -> answer", run_mode="unrestricted")
r_md = react(question="When did I take the 3 latest Screenshots on this computer (look in the screenshots folder within pictures)?")
print(r_md)

# Prediction(
#     trajectory={'thought_0': 'To determine when the latest screenshots were taken, I need to access the file system to check the modification or creation dates of the screenshot files in the Screenshots folder within the Pictures directory. I will use Python to achieve this.', 'assistant_response_0': '```python\n# | run\nimport os\nfrom datetime import datetime\n\n# Define the path to the Screenshots folder\nscreenshots_folder = os.path.join(os.path.expanduser(\'~\'), \'Pictures\', \'Screenshots\')\n\n# Check if the Screenshots folder exists\nif os.path.exists(screenshots_folder):\n    # Get a list of all files in the Screenshots folder\n    files = [os.path.join(screenshots_folder, f) for f in os.listdir(screenshots_folder)]\n\n    # Filter out directories and get the modification times\n    files_with_mtimes = [(f, os.path.getmtime(f)) for f in files if os.path.isfile(f)]\n\n    # Sort the files by their modification times\n    files_with_mtimes.sort(key=lambda x: x[1], reverse=True)\n\n    # Get the latest modification times\n    latest_screenshots = files_with_mtimes[:3]\n\n    # Convert the modification times to human-readable format\n    latest_dates = [datetime.fromtimestamp(mtime).strftime(\'%Y-%m-%d %H:%M:%S\') for _, mtime in latest_screenshots]\n\n    # Print the dates of the latest screenshots\n    print("The latest screenshots were taken on:")\n    for date in latest_dates:\n        print(date)\nelse:\n    print("The Screenshots folder does not exist.")\n```', 'observation_2': '```python\n# | result\ncode_block -> The latest screenshots were taken on:\n2025-06-17 22:33:18\n2025-06-17 22:27:44\n2025-06-17 22:22:40\n```', 'thought_1': 'I have successfully retrieved the dates of the latest screenshots. Now, I will format this information into a clear and concise answer for the user.', 'assistant_response_1': 'The latest screenshots on this computer were taken on the following dates:\n1. 2025-06-17 22:33:18\n2. 2025-06-17 22:27:44\n3. 2025-06-17 22:22:40'},
#     reasoning='To determine when the latest screenshots were taken, I accessed the file system to check the modification dates of the screenshot files in the Screenshots folder within the Pictures directory. Using Python, I listed the files in the Screenshots folder, filtered out directories, and sorted the files by their modification times. The three most recent modification dates were then converted to a human-readable format and presented.',
#     answer='The latest screenshots on this computer were taken on the following dates:\n1. 2025-06-17 22:33:18\n2. 2025-06-17 22:27:44\n3. 2025-06-17 22:22:40'
# )
