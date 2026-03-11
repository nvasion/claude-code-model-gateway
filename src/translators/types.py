"""Canonical request/response types for model provider translation.

The canonical format follows the OpenAI Chat Completions API schema, which
is the most widely adopted interface in the AI ecosystem. All provider
translators convert to/from this format.

Canonical request structure::

    {
        "model": "gpt-4o",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there!"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is in this image?"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}},
                ],
            },
        ],
        "max_tokens": 1024,
        "temperature": 0.7,
        "stream": false,
        "tools": [...],
        "tool_choice": "auto",
    }

Canonical response structure::

    {
        "id": "chatcmpl-xxx",
        "object": "chat.completion",
        "created": 1712345678,
        "model": "gpt-4o",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Response text",
                    "tool_calls": [...],
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30,
        },
    }
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Union

try:
    from typing import TypedDict
except ImportError:  # Python < 3.8
    from typing_extensions import TypedDict  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Message content types
# ---------------------------------------------------------------------------


class TextContentPart(TypedDict, total=False):
    """A plain-text content part inside a multimodal message."""

    type: Literal["text"]
    text: str


class ImageUrl(TypedDict, total=False):
    """URL or base64 data-URI for an inline image."""

    url: str
    detail: Literal["auto", "low", "high"]


class ImageUrlContentPart(TypedDict, total=False):
    """An image content part inside a multimodal message."""

    type: Literal["image_url"]
    image_url: ImageUrl


# Union of all supported content-part types
ContentPart = Union[TextContentPart, ImageUrlContentPart, Dict[str, Any]]

# Message content is either a plain string or a list of content parts.
MessageContent = Union[str, List[ContentPart]]


# ---------------------------------------------------------------------------
# Tool / function calling types
# ---------------------------------------------------------------------------


class FunctionParameters(TypedDict, total=False):
    """JSON Schema object describing a function's parameters."""

    type: str
    properties: Dict[str, Any]
    required: List[str]
    additionalProperties: bool


class FunctionDefinition(TypedDict, total=False):
    """Schema for a single callable function."""

    name: str
    description: str
    parameters: FunctionParameters


class Tool(TypedDict, total=False):
    """A tool (function) that the model can call."""

    type: Literal["function"]
    function: FunctionDefinition


class ToolCallFunction(TypedDict, total=False):
    """The function invocation inside a :class:`ToolCall`."""

    name: str
    arguments: str  # JSON-encoded string


class ToolCall(TypedDict, total=False):
    """A single tool-call request produced by the model."""

    id: str
    type: Literal["function"]
    function: ToolCallFunction


# ---------------------------------------------------------------------------
# Message types
# ---------------------------------------------------------------------------


class SystemMessage(TypedDict, total=False):
    """System-level instruction message."""

    role: Literal["system"]
    content: str
    name: str


class UserMessage(TypedDict, total=False):
    """User-turn message (text or multimodal)."""

    role: Literal["user"]
    content: MessageContent
    name: str


class AssistantMessage(TypedDict, total=False):
    """Assistant-turn message, optionally containing tool calls."""

    role: Literal["assistant"]
    content: Optional[MessageContent]
    name: str
    tool_calls: List[ToolCall]


class ToolMessage(TypedDict, total=False):
    """The result of a tool call returned by the caller."""

    role: Literal["tool"]
    content: str
    tool_call_id: str
    name: str


# Union of all message types
Message = Union[SystemMessage, UserMessage, AssistantMessage, ToolMessage, Dict[str, Any]]


# ---------------------------------------------------------------------------
# Canonical request
# ---------------------------------------------------------------------------


class CanonicalRequest(TypedDict, total=False):
    """Canonical request in OpenAI Chat Completions format.

    Required keys: ``model``, ``messages``.
    All other keys are optional.
    """

    # --- Required ---
    model: str
    messages: List[Message]

    # --- Sampling ---
    max_tokens: int
    temperature: float
    top_p: float
    n: int
    seed: int
    stop: Union[str, List[str]]

    # --- Penalties ---
    presence_penalty: float
    frequency_penalty: float
    logit_bias: Dict[str, float]

    # --- Logging ---
    logprobs: bool
    top_logprobs: int

    # --- Streaming ---
    stream: bool
    stream_options: Dict[str, Any]

    # --- Tools ---
    tools: List[Tool]
    tool_choice: Union[Literal["none", "auto", "required"], Dict[str, Any]]
    parallel_tool_calls: bool

    # --- Format ---
    response_format: Dict[str, Any]

    # --- Metadata ---
    user: str
    metadata: Dict[str, Any]


# ---------------------------------------------------------------------------
# Canonical response
# ---------------------------------------------------------------------------


class UsageInfo(TypedDict, total=False):
    """Token-usage statistics for a completion request."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ResponseMessage(TypedDict, total=False):
    """The assistant message inside a :class:`Choice`."""

    role: str
    content: Optional[str]
    tool_calls: List[ToolCall]
    # Deprecated; kept for compatibility with older OpenAI responses
    function_call: Dict[str, str]


class Choice(TypedDict, total=False):
    """A single completion alternative."""

    index: int
    message: ResponseMessage
    finish_reason: Optional[str]
    logprobs: Optional[Dict[str, Any]]


class CanonicalResponse(TypedDict, total=False):
    """Canonical response in OpenAI Chat Completions format."""

    id: str
    object: Literal["chat.completion"]
    created: int
    model: str
    choices: List[Choice]
    usage: UsageInfo
    system_fingerprint: Optional[str]


# ---------------------------------------------------------------------------
# Streaming chunk types
# ---------------------------------------------------------------------------


class StreamDelta(TypedDict, total=False):
    """Incremental delta content in a streaming chunk."""

    role: str
    content: Optional[str]
    tool_calls: List[Dict[str, Any]]


class StreamChoice(TypedDict, total=False):
    """A single streaming choice delta."""

    index: int
    delta: StreamDelta
    finish_reason: Optional[str]
    logprobs: Optional[Dict[str, Any]]


class CanonicalStreamChunk(TypedDict, total=False):
    """Canonical Server-Sent Events chunk in OpenAI streaming format.

    Emitted repeatedly during a streaming completion, terminated by a
    ``data: [DONE]`` sentinel (not represented as a chunk).
    """

    id: str
    object: Literal["chat.completion.chunk"]
    created: int
    model: str
    choices: List[StreamChoice]
    # Only present on the final chunk when stream_options.include_usage=True
    usage: Optional[UsageInfo]


# ---------------------------------------------------------------------------
# Finish-reason constants (canonical)
# ---------------------------------------------------------------------------


class FinishReason:
    """Canonical finish-reason string constants."""

    STOP: Literal["stop"] = "stop"
    LENGTH: Literal["length"] = "length"
    TOOL_CALLS: Literal["tool_calls"] = "tool_calls"
    CONTENT_FILTER: Literal["content_filter"] = "content_filter"
    NULL: None = None
