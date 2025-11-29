# llm_api/openrouter_api.py
from __future__ import annotations
import os, time, random, json
from typing import Optional, Dict, Any, List
import requests
from dotenv import load_dotenv
load_dotenv()

OPENROUTER_API_KEY_ENV = "OPENROUTER_API_KEY"
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

DEFAULT_MODEL = os.getenv("OPENROUTER_MODEL", "mistralai/mistral-small-3.1-24b-instruct:free")

# 🔥 可配置的超时和重试参数
DEFAULT_TIMEOUT = float(os.getenv("OPENROUTER_TIMEOUT", "5.0"))  # 默认 5 秒
DEFAULT_MAX_RETRIES = int(os.getenv("OPENROUTER_MAX_RETRIES", "3"))  # 默认重试 3 次
RETRY_BACKOFF_BASE = float(os.getenv("OPENROUTER_RETRY_BACKOFF", "2.0"))  # 指数退避基数

# ======= Provider 配置（从 .env 读取）=======
def get_provider_config() -> Optional[Dict[str, Any]]:
    """
    从环境变量读取 provider 配置。
    如果未指定，返回 None（使用 OpenRouter 默认路由）。
    """
    provider = os.getenv("OPENROUTER_PROVIDER", "").strip()
    
    if not provider:
        return None
    
    # 指定单个 provider，不允许 fallback
    return {
        "order": [provider],
        "allow_fallbacks": False,
    }

SYSTEM_PROMPT = (
    "You are an IELTS Writing Task 2 examiner. "
    "Be strict but fair. "
    "Output ONLY a single final overall band score (0-9 in 0.5 steps). "
    "No explanation. No extra words."
)

# --- 简单 circuit breaker: 连续上游429 -> 暂停一会再试 ---
_CB_STATE = {
    "blocked_until": 0.0,
    "consec_429": 0,
}



def _headers(api_key: str):
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        # referer/title 可留可不留，但留着更稳
        "HTTP-Referer": os.getenv("OPENROUTER_REFERER", "https://ielts-alphaevolve.local"),
        "X-Title": os.getenv("OPENROUTER_TITLE", "IELTS-AlphaEvolve"),
    }

def _payload(use_model: str, prompt: str, temperature: float, max_tokens: int):
    # 只放最小兼容字段，避免 provider 404/400
    return {
        "model": use_model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": float(temperature),
        "top_p": 1.0,
        "max_tokens": int(max_tokens),
        "stream": False,
    }

# llm_api/openrouter_api.py


def call_scoring_llm(
    prompt: str,
    temperature: float = 0.0,
    max_retries: Optional[int] = None,
    timeout: Optional[float] = None,
    model: Optional[str] = None,
    max_tokens: int = 8,
) -> str:
    """
    调用 OpenRouter API 进行评分。
    
    Args:
        prompt: 输入的 prompt
        temperature: 温度参数
        max_retries: 最大重试次数（None 则使用环境变量配置）
        timeout: 超时时间（秒，None 则使用环境变量配置）
        model: 模型名称
        max_tokens: 最大 token 数
    
    Returns:
        LLM 返回的文本，失败返回空字符串
    """
    api_key = os.getenv(OPENROUTER_API_KEY_ENV)
    if not api_key:
        raise RuntimeError(f"Please set env {OPENROUTER_API_KEY_ENV}")

    use_model = model or DEFAULT_MODEL
    use_timeout = timeout if timeout is not None else DEFAULT_TIMEOUT
    use_max_retries = max_retries if max_retries is not None else DEFAULT_MAX_RETRIES

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://ielts-alphaevolve.local",
        "X-Title": "IELTS-AlphaEvolve",
    }

    # 从 .env 读取 provider 配置
    provider_config = get_provider_config()

    payload = {
        "model": use_model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": prompt
                           + "\n\nReturn ONLY the final overall band score (one number)."
            },
        ],
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
    }
    
    # 只有配置了 provider 才添加到 payload
    if provider_config:
        payload["provider"] = provider_config

    # circuit breaker：如果刚被连续429打爆，直接返回空（让上层缺省或 early-stop）
    now = time.time()
    if now < _CB_STATE["blocked_until"]:
        print(f"[OpenRouter] {use_model} circuit-breaker active, skip call.")
        return ""

    last_err = None
    for attempt in range(1, use_max_retries + 1):
        try:
            # 🔥 使用配置的超时时间
            start_time = time.time()
            r = requests.post(
                OPENROUTER_API_URL, 
                headers=headers, 
                json=payload, 
                timeout=use_timeout
            )
            elapsed = time.time() - start_time

            # 处理各种 HTTP 状态码
            if r.status_code in (402, 500, 502, 503):
                backoff = min(RETRY_BACKOFF_BASE ** attempt + random.random(), 15)
                print(f"[OpenRouter] {use_model} -> {r.status_code} (attempt {attempt}/{use_max_retries}), backoff {backoff:.1f}s")
                time.sleep(backoff)
                last_err = r.text
                continue

            if r.status_code == 429:
                _CB_STATE["consec_429"] += 1
                backoff = min(RETRY_BACKOFF_BASE ** attempt + random.random() * 1.5, 20)
                print(f"[OpenRouter] {use_model} -> 429 (attempt {attempt}/{use_max_retries}), backoff {backoff:.1f}s")
                time.sleep(backoff)
                last_err = r.text

                # 连续 3 次 429：触发短路
                if _CB_STATE["consec_429"] >= 3:
                    _CB_STATE["blocked_until"] = time.time() + 60
                    print(f"[OpenRouter] {use_model} upstream 429 streak -> open circuit for 60s.")
                    break
                continue

            r.raise_for_status()
            data = r.json()
            text = (data["choices"][0]["message"]["content"] or "").strip()

            # 成功就重置 429 streak
            _CB_STATE["consec_429"] = 0
            print(f"[OpenRouter] ✅ {use_model} responded in {elapsed:.2f}s")
            return text

        except requests.exceptions.Timeout:
            # 🔥 超时处理
            last_err = f"Timeout after {use_timeout}s"
            print(f"[OpenRouter] ⏱️  {use_model} timeout ({use_timeout}s) on attempt {attempt}/{use_max_retries}")
            
            # 超时后立即重试，不等待
            if attempt < use_max_retries:
                print(f"[OpenRouter] 🔄 Retrying immediately...")
                continue
            else:
                print(f"[OpenRouter] ❌ Max retries reached after timeout")
                break
        
        except requests.exceptions.RequestException as e:
            # 🔥 网络错误处理
            last_err = str(e)
            backoff = min(RETRY_BACKOFF_BASE ** attempt + random.random(), 10)
            print(f"[OpenRouter] 🌐 {use_model} network error: {e} (attempt {attempt}/{use_max_retries}), backoff {backoff:.1f}s")
            time.sleep(backoff)
            continue
        
        except Exception as e:
            # 🔥 其他异常
            last_err = str(e)
            backoff = min(RETRY_BACKOFF_BASE ** attempt + random.random(), 10)
            print(f"[OpenRouter] ⚠️  {use_model} exception: {e} (attempt {attempt}/{use_max_retries}), backoff {backoff:.1f}s")
            time.sleep(backoff)
            continue

    print(f"[OpenRouter] ❌ All retries failed for {use_model}, last_err={last_err}")
    return ""
