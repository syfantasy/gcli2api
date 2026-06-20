"""
Request Sanitizer - 入站请求体容错清洗
────────────────────────────────────────────────────────────────
针对部分第三方客户端（如某些机器人插件）发包不规范的问题，在请求体
进入 Pydantic 模型验证之前做容错清洗，避免本可挽救的请求被拦下报 422。

处理两类 Bug：

1) 双重序列化 (Double Stringification)
   客户端把整个 JSON 对象又 stringify 了一层，导致请求体实际是一个带
   转义的 JSON 字符串：
       "{\\"contents\\":[{\\"parts\\":[{\\"text\\":\\"hi\\"}]}]}"
   Pydantic 会报 422: "Input should be a valid dictionary or object ..."
   => 自动 json.loads 解包回 dict（兼容意外的多层包裹）。

2) 缺失必填的 role 字段
   Gemini 规范要求 contents[] 每个对象都带 role，但客户端常常漏传。
   Pydantic 会报 422: "contents.0.role Field required"
   => 自动为缺失 role 的 content 补齐 "role": "user"。

实现方式采用 FastAPI 官方推荐的「自定义 Request / APIRoute」模式
（参见 https://fastapi.tiangolo.com/how-to/custom-request-and-route/）：
重写 Request.body() 返回清洗后的字节流。FastAPI 在做 body 校验时会先
`await request.body()` 再 `await request.json()`，而 Starlette 的
`json()` 又是基于 `body()` 实现的，因此清洗结果会透明地流入后续的
Pydantic 验证，无需改动任何数据模型。

仅挂载到 Gemini 原生格式路由（geminicli / antigravity），不影响
OpenAI / Anthropic / 面板等其它路由；对格式良好的请求是 no-op。
"""

import json
from typing import Any, Optional, Tuple

from fastapi import Request
from fastapi.routing import APIRoute

from log import log

# 双重序列化最多解包层数：防御异常的多层包裹，同时避免极端情况下的死循环
_MAX_UNWRAP_DEPTH = 3
# 缺失 role 时默认补齐的角色
_DEFAULT_ROLE = "user"


def _clean_content_item(item: dict) -> Tuple[dict, bool]:
    """
    清洗单个 content 字典对象，为其补齐 role 并处理潜在的 parts 双重序列化。
    """
    changed = False
    
    # 兼容 role 缺失、为 None、或为空字符串的情况
    role = item.get("role")
    if not role or not isinstance(role, str) or role.strip() == "":
        item["role"] = _DEFAULT_ROLE
        changed = True
        
    # 处理 parts 本身被序列化为字符串的情况
    parts = item.get("parts")
    if isinstance(parts, str):
        try:
            parsed_parts = json.loads(parts)
            if isinstance(parsed_parts, list):
                item["parts"] = parsed_parts
                parts = parsed_parts
                changed = True
        except Exception:
            pass
            
    # 清洗 parts 列表中的子项
    if isinstance(parts, list):
        new_parts = []
        for part in parts:
            if isinstance(part, str):
                try:
                    parsed_part = json.loads(part)
                    if isinstance(parsed_part, dict):
                        part = parsed_part
                        changed = True
                except Exception:
                    pass
            new_parts.append(part)
        item["parts"] = new_parts
        
    return item, changed


def sanitize_gemini_payload(data: Any) -> Tuple[Any, bool]:
    """
    对已解析为 Python 对象的请求体做就地容错清洗。

    纯函数（不涉及 I/O），便于单元测试与复用。

    Args:
        data: json.loads 后的对象（可能是 dict，也可能因双重序列化而是 str）

    Returns:
        (cleaned_obj, changed): 清洗后的对象，以及是否发生了实际修改
    """
    changed = False

    # ---------- Bug 1: 双重序列化解包 ----------
    # 如果对象本身是 str，说明客户端把 JSON 又 stringify 了一层（可能多层）。
    depth = 0
    while isinstance(data, str) and depth < _MAX_UNWRAP_DEPTH:
        try:
            data = json.loads(data)
            changed = True
        except (json.JSONDecodeError, ValueError):
            # 尝试剥离外层引号，处理带有不规范转义的字符串
            stripped = data.strip()
            if (stripped.startswith('"') and stripped.endswith('"')) or (stripped.startswith("'") and stripped.endswith("'")):
                try:
                    data = json.loads(stripped[1:-1].replace('\\"', '"').replace('\\\\', '\\'))
                    changed = True
                except Exception:
                    break
            else:
                break
        depth += 1

    # ---------- Bug 2: 清洗 contents 列表与补齐 role ----------
    if isinstance(data, dict):
        contents = data.get("contents")
        
        # 处理 contents 被意外序列化为字符串的情况
        if isinstance(contents, str):
            try:
                parsed_contents = json.loads(contents)
                if isinstance(parsed_contents, list):
                    data["contents"] = parsed_contents
                    contents = parsed_contents
                    changed = True
            except Exception:
                pass
                
        # 遍历清洗 contents 列表中的每一项
        if isinstance(contents, list):
            new_contents = []
            for item in contents:
                # 处理 contents 中的元素本身被序列化为字符串的情况
                if isinstance(item, str):
                    try:
                        parsed_item = json.loads(item)
                        if isinstance(parsed_item, dict):
                            item = parsed_item
                            changed = True
                        elif isinstance(parsed_item, list):
                            # 如果是一个序列化的列表，将其平铺展开
                            for sub_item in parsed_item:
                                if isinstance(sub_item, dict):
                                    cleaned_sub, sub_changed = _clean_content_item(sub_item)
                                    new_contents.append(cleaned_sub)
                                    if sub_changed:
                                        changed = True
                            changed = True
                            continue
                    except Exception:
                        pass
                
                if isinstance(item, dict):
                    cleaned_item, item_changed = _clean_content_item(item)
                    new_contents.append(cleaned_item)
                    if item_changed:
                        changed = True
                else:
                    new_contents.append(item)
            
            data["contents"] = new_contents

    return data, changed


def _sanitize_body_bytes(raw: bytes) -> Optional[bytes]:
    """
    清洗原始请求体字节流。

    Returns:
        清洗后的新字节流；若无需修改 / 无法解析则返回 None
        （表示沿用原始字节，让 FastAPI 走它原本的解析与错误处理）。
    """
    if not raw:
        return None

    # 解析原始 JSON；失败则交回原样（不是 JSON 的请求不归我们管）
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, ValueError, UnicodeDecodeError):
        # 回退处理：如果原始字节流本身是格式不规范的转义 JSON 字符串
        try:
            raw_str = raw.decode("utf-8").strip()
            if (raw_str.startswith('"') and raw_str.endswith('"')) or (raw_str.startswith("'") and raw_str.endswith("'")):
                unescaped = raw_str[1:-1].replace('\\"', '"').replace('\\\\', '\\')
                data = json.loads(unescaped)
            else:
                return None
        except Exception:
            return None

    cleaned, changed = sanitize_gemini_payload(data)
    if not changed:
        return None

    try:
        # ensure_ascii=False 保留原始 UTF-8 文本，避免中文等被转义膨胀
        return json.dumps(cleaned, ensure_ascii=False).encode("utf-8")
    except (TypeError, ValueError):
        return None


class _SanitizedRequest(Request):
    """重写 body()，在首次读取并缓存进 _body 之前完成清洗。"""

    async def body(self) -> bytes:
        if not hasattr(self, "_body"):
            raw = await super().body()
            cleaned: Optional[bytes]
            try:
                cleaned = _sanitize_body_bytes(raw)
            except Exception as exc:  # 清洗逻辑本身绝不能让正常请求挂掉
                log.warning(f"[SANITIZER] 请求体清洗异常，按原样放行: {exc}")
                cleaned = None

            if cleaned is not None:
                log.debug("[SANITIZER] 已清洗入站 Gemini 请求体（解包/补齐 role）")
                self._body = cleaned
            else:
                self._body = raw
        return self._body


class GeminiSanitizeRoute(APIRoute):
    """
    自定义路由类：把请求包装为 _SanitizedRequest 后再交给原始处理器。

    用法：在创建路由器时传入 route_class 即可对该路由器下所有路由生效：
        router = APIRouter(route_class=GeminiSanitizeRoute)
    """

    def get_route_handler(self):
        original_route_handler = super().get_route_handler()

        async def custom_route_handler(request: Request):
            request = _SanitizedRequest(request.scope, request.receive)
            return await original_route_handler(request)

        return custom_route_handler
