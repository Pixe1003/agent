"""验证 agent_sft._parse_tool_call 对模型输出的鲁棒性。

不依赖真实 GGUF 模型 — 只测纯解析逻辑。
"""
from agent_sft.scheduler import _parse_tool_call


def test_canonical_tool_call_with_close_tag():
    text = '<tool_call>\n{"name": "select_server", "arguments": {"server_id": 3, "reasoning": "best fit"}}\n</tool_call>'
    out = _parse_tool_call(text)
    assert out is not None
    assert out["name"] == "select_server"
    assert out["arguments"]["server_id"] == 3


def test_tool_call_without_close_tag():
    """stop token 截断时 </tool_call> 可能缺失。"""
    text = '<tool_call>\n{"name": "select_server", "arguments": {"server_id": 1, "reasoning": "ok"}}'
    out = _parse_tool_call(text)
    assert out is not None
    assert out["name"] == "select_server"


def test_reject_service_tool_call():
    text = '<tool_call>{"name": "reject_service", "arguments": {"reason": "no fit"}}</tool_call>'
    out = _parse_tool_call(text)
    assert out is not None
    assert out["name"] == "reject_service"
    assert "reason" in out["arguments"]


def test_bare_json_without_tag():
    """有时候模型直接吐 JSON 不加 tag。"""
    text = '{"name": "select_server", "arguments": {"server_id": 0, "reasoning": "ok"}}'
    out = _parse_tool_call(text)
    assert out is not None
    assert out["arguments"]["server_id"] == 0


def test_nested_braces_in_reasoning():
    text = '<tool_call>{"name": "select_server", "arguments": {"server_id": 2, "reasoning": "spread {balanced}"}}</tool_call>'
    out = _parse_tool_call(text)
    assert out is not None
    assert "{balanced}" in out["arguments"]["reasoning"]


def test_garbage_text_returns_none():
    assert _parse_tool_call("I think server 3 is best") is None
    assert _parse_tool_call("") is None
    assert _parse_tool_call("<tool_call>not json</tool_call>") is None


def test_unknown_tool_name_returns_none():
    text = '<tool_call>{"name": "drop_service", "arguments": {}}</tool_call>'
    assert _parse_tool_call(text) is None


def test_missing_arguments_returns_none():
    text = '<tool_call>{"name": "select_server"}</tool_call>'
    assert _parse_tool_call(text) is None


def test_arguments_not_dict_returns_none():
    text = '<tool_call>{"name": "select_server", "arguments": "server 1"}</tool_call>'
    assert _parse_tool_call(text) is None
