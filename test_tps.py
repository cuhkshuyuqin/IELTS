#!/usr/bin/env python
"""
测试 OpenRouter 配置的 Tokens Per Second (TPS)
"""
import os
import time
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
MODEL = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct")
PROVIDER = os.getenv("OPENROUTER_PROVIDER", "").strip()

if not API_KEY:
    print("❌ 请在 .env 中设置 OPENROUTER_API_KEY")
    exit(1)

BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
    "HTTP-Referer": "https://ielts-alphaevolve.local",
    "X-Title": "IELTS-TPS-Test",
}

# 测试 prompt（让模型生成一段文本）
test_prompt = """Write a detailed explanation of how photosynthesis works in plants, 
including the light-dependent and light-independent reactions. Be thorough and scientific."""

def build_payload(max_tokens: int = 500):
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "user", "content": test_prompt}
        ],
        "temperature": 0.7,
        "max_tokens": max_tokens,
    }
    
    if PROVIDER:
        payload["provider"] = {
            "order": [PROVIDER],
            "allow_fallbacks": False,
        }
    
    return payload

def test_tps(num_tests: int = 3, max_tokens: int = 500):
    print("=" * 60)
    print("🧪 OpenRouter TPS 测试")
    print("=" * 60)
    print(f"模型: {MODEL}")
    print(f"Provider: {PROVIDER if PROVIDER else '默认路由'}")
    print(f"Max Tokens: {max_tokens}")
    print(f"测试次数: {num_tests}")
    print("=" * 60)
    
    results = []
    
    for i in range(1, num_tests + 1):
        print(f"\n📊 测试 {i}/{num_tests}...")
        
        payload = build_payload(max_tokens)
        
        try:
            start_time = time.time()
            response = requests.post(BASE_URL, headers=headers, json=payload, timeout=120)
            end_time = time.time()
            
            if response.status_code != 200:
                print(f"❌ 错误: {response.status_code}")
                print(f"响应: {response.text[:300]}")
                continue
            
            data = response.json()
            
            # 提取信息
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            usage = data.get("usage", {})
            
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            total_tokens = usage.get("total_tokens", 0)
            
            elapsed_time = end_time - start_time
            
            # 计算 TPS
            tps = completion_tokens / elapsed_time if elapsed_time > 0 else 0
            
            results.append({
                "test": i,
                "elapsed_time": elapsed_time,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "tps": tps,
                "content_length": len(content),
            })
            
            print(f"✅ 完成")
            print(f"   耗时: {elapsed_time:.2f}s")
            print(f"   Prompt tokens: {prompt_tokens}")
            print(f"   Completion tokens: {completion_tokens}")
            print(f"   Total tokens: {total_tokens}")
            print(f"   TPS: {tps:.2f} tokens/s")
            print(f"   生成内容长度: {len(content)} 字符")
            
            # 避免触发限流
            if i < num_tests:
                print("   等待 3 秒...")
                time.sleep(3)
        
        except requests.exceptions.Timeout:
            print(f"❌ 超时")
        except Exception as e:
            print(f"❌ 异常: {e}")
    
    # 统计结果
    if results:
        print("\n" + "=" * 60)
        print("📈 统计结果")
        print("=" * 60)
        
        avg_time = sum(r["elapsed_time"] for r in results) / len(results)
        avg_completion = sum(r["completion_tokens"] for r in results) / len(results)
        avg_tps = sum(r["tps"] for r in results) / len(results)
        min_tps = min(r["tps"] for r in results)
        max_tps = max(r["tps"] for r in results)
        
        print(f"平均耗时: {avg_time:.2f}s")
        print(f"平均生成 tokens: {avg_completion:.1f}")
        print(f"平均 TPS: {avg_tps:.2f} tokens/s")
        print(f"最小 TPS: {min_tps:.2f} tokens/s")
        print(f"最大 TPS: {max_tps:.2f} tokens/s")
        print("=" * 60)
        
        # 性能评估
        print("\n💡 性能评估:")
        if avg_tps >= 50:
            print("   🚀 优秀 (≥50 TPS)")
        elif avg_tps >= 30:
            print("   ✅ 良好 (30-50 TPS)")
        elif avg_tps >= 15:
            print("   ⚠️  一般 (15-30 TPS)")
        else:
            print("   🐌 较慢 (<15 TPS)")
    else:
        print("\n❌ 没有成功的测试结果")

if __name__ == "__main__":
    import sys
    
    # 可以通过命令行参数指定测试次数和 max_tokens
    num_tests = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    max_tokens = int(sys.argv[2]) if len(sys.argv) > 2 else 500
    
    test_tps(num_tests=num_tests, max_tokens=max_tokens)
