"""GeminiController: implements the Controller Protocol required by agentic.py.

The controller calls the Gemini API with a `search` function-call tool.
On each turn it receives:
  - question  : the original user question
  - history   : list of dicts built by run_agent
                [{'tool': 'search', 'queries': [...], 'context': str, 'context_ids': [...]}]

It returns one of:
  {'action': 'search', 'query': '...'}
  {'action': 'answer', 'answer': '...', 'citations': ['chunk-id', ...]}

Citations must appear as [chunk-id] inside the answer text; run_agent enforces
this. The controller never receives gold answers; it works only from context.
"""
from __future__ import annotations

import json
import re
from typing import Any

import google.genai as genai
import google.genai.types as gtypes


# ---------------------------------------------------------------------------
# System prompt shown to Gemini on every turn
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT = """Bạn là trợ lý trả lời câu hỏi dựa trên tài liệu.

Bạn có một công cụ: `search(query)`.
- Dùng `search` khi cần tìm bằng chứng từ tài liệu.
- Khi đã có đủ bằng chứng trong context, trả lời bằng JSON sau (không có markdown):
  {"action": "answer", "answer": "<câu trả lời, chèn [chunk-id] cho mỗi chunk dùng làm bằng chứng>", "citations": ["chunk-id", ...]}
- Nếu không tìm thấy bằng chứng sau khi tìm kiếm, trả lời:
  {"action": "answer", "answer": "Không đủ bằng chứng để trả lời.", "citations": []}

Quy tắc bắt buộc:
1. Mỗi citation trong "citations" PHẢI xuất hiện dạng [chunk-id] trong "answer".
2. Không bịa đặt thông tin ngoài context.
3. Không đưa nội dung gold hay đáp án chuẩn vào câu trả lời.
4. Chỉ dùng chunk-id thực tế từ context (dạng như "policy-1:p1:c1").
5. Mỗi chunk-id phải nằm trong cặp ngoặc vuông RIÊNG, ví dụ "[id-1] [id-2]". KHÔNG gộp nhiều id trong một cặp ngoặc như "[id-1, id-2]".
"""

# ---------------------------------------------------------------------------
# Tool definition for Gemini function-calling
# ---------------------------------------------------------------------------
_SEARCH_TOOL = gtypes.Tool(
    function_declarations=[
        gtypes.FunctionDeclaration(
            name="search",
            description="Tìm các đoạn văn bản liên quan từ tài liệu theo câu query.",
            parameters=gtypes.Schema(
                type=gtypes.Type.OBJECT,
                properties={
                    "query": gtypes.Schema(
                        type=gtypes.Type.STRING,
                        description="Câu query ngắn gọn để tìm kiếm bằng chứng.",
                    )
                },
                required=["query"],
            ),
        )
    ]
)


# ---------------------------------------------------------------------------
# Helper: build Gemini message list from run_agent history
# ---------------------------------------------------------------------------
_NO_REPEAT_HINT = (
    "- Tuyệt đối KHÔNG lặp lại query đã dùng (xem danh sách trên). Nếu cần tìm thêm, hãy đổi hẳn "
    "từ khóa hoặc cách diễn đạt: dùng từ đồng nghĩa, tên mục, tên bảng hoặc thuật ngữ khác "
    "(ví dụ chuyển từ câu hỏi sang cụm từ khóa ngắn).\n"
)


def _build_messages(
    question: str,
    history: list[dict[str, Any]],
    no_repeat: bool = False,
) -> list[gtypes.Content]:
    """Convert run_agent history format to Gemini Content list.

    To avoid thought_signature requirements on synthetic functionCall parts,
    retrieved context from previous turns is presented cleanly in the user prompt.
    """
    if not history:
        return [
            gtypes.Content(
                role="user",
                parts=[gtypes.Part(text=f"Câu hỏi: {question}")],
            )
        ]

    latest_context = history[-1].get("context", "")
    context_ids = history[-1].get("context_ids", [])
    queries = history[-1].get("queries", [])

    prompt_text = (
        f"Câu hỏi: {question}\n\n"
        f"Các truy vấn đã tìm kiếm trước đó: {queries}\n\n"
        f"Tài liệu trích xuất (Context):\n{latest_context}\n\n"
        f"Các Chunk IDs có trong context: {context_ids}\n\n"
        "Hướng dẫn:\n"
        "- Nếu context đã chứa đủ bằng chứng để trả lời, hãy trả lời đầy đủ và bắt buộc chèn [chunk-id] "
        "cho mỗi bằng chứng được sử dụng (ví dụ: '... 7 ngày làm việc [policy-1:p1:c1].').\n"
        "- Khi trích nhiều chunk, mỗi chunk-id phải nằm trong cặp ngoặc vuông riêng: '[id-1] [id-2]', không viết '[id-1, id-2]'.\n"
        "- Nếu context chưa đủ và cần tìm kiếm thêm thông tin khác, hãy gọi công cụ search(query).\n"
        + (_NO_REPEAT_HINT if no_repeat else "")
        + "- Nếu tài liệu không chứa câu trả lời, hãy trả lời: 'Không đủ bằng chứng để trả lời.' với citations rỗng []."
    )

    return [
        gtypes.Content(
            role="user",
            parts=[gtypes.Part(text=prompt_text)],
        )
    ]


# ---------------------------------------------------------------------------
# Helper: parse model text response into action dict
# ---------------------------------------------------------------------------
def _parse_text_response(text: str) -> dict[str, Any]:
    """Extract JSON action from model text; fallback to abstain."""
    # Try to find JSON block (with or without markdown fences)
    json_match = re.search(r'\{.*?"action"\s*:.*?\}', text, re.DOTALL)
    if json_match:
        try:
            parsed = json.loads(json_match.group())
            if parsed.get("action") in ("search", "answer"):
                return parsed
        except json.JSONDecodeError:
            pass
    # Fallback: treat the whole text as an answer without citations
    return {
        "action": "answer",
        "answer": text.strip() or "Không đủ bằng chứng để trả lời.",
        "citations": [],
    }


# =============================================================================
# GeminiController
# =============================================================================

class GeminiController:
    """Calls Gemini with function-calling to decide search vs answer.

    Args:
        api_key: Gemini API key (from GEMINI_API_KEY in .env).
        model:   Gemini model name. Default is gemini-2.0-flash.
        temperature: Sampling temperature (lower = more deterministic).
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-3.5-flash-lite",
        temperature: float = 0.0,
        no_repeat: bool = False,
    ) -> None:
        self._client = genai.Client(api_key=api_key)
        self._model = model
        self._temperature = temperature
        self._no_repeat = no_repeat

    # ------------------------------------------------------------------
    def decide(
        self,
        question: str,
        history: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Call Gemini with automatic retry on rate limits, and return action dict."""
        import time

        messages = _build_messages(question, history, self._no_repeat)

        for attempt in range(4):
            try:
                response = self._client.models.generate_content(
                    model=self._model,
                    contents=messages,
                    config=gtypes.GenerateContentConfig(
                        system_instruction=_SYSTEM_PROMPT,
                        tools=[_SEARCH_TOOL],
                        temperature=self._temperature,
                        tool_config=gtypes.ToolConfig(
                            function_calling_config=gtypes.FunctionCallingConfig(
                                mode=gtypes.FunctionCallingConfigMode.AUTO,
                            )
                        ),
                    ),
                )
                break
            except Exception as e:
                err_str = str(e)
                if ("429" in err_str or "RESOURCE_EXHAUSTED" in err_str) and attempt < 3:
                    wait_sec = 25 * (attempt + 1)
                    print(f"       [Rate limit 429] Waiting {wait_sec}s before retry attempt {attempt + 1}...", flush=True)
                    time.sleep(wait_sec)
                else:
                    raise

        candidate = response.candidates[0]
        content = candidate.content

        # Case 1: model chose to call the search function
        for part in content.parts:
            if part.function_call and part.function_call.name == "search":
                query = (part.function_call.args or {}).get("query", "").strip()
                if query:
                    return {"action": "search", "query": query}

        # Case 2: model produced a text response
        text = "".join(
            part.text for part in content.parts if hasattr(part, "text") and part.text
        )
        return _parse_text_response(text)
