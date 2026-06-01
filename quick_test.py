"""
快速测试脚本 - 验证 Qwen2.5 能否正常调用
"""

from ollama import Client

def test_ollama_connection():
    """测试 Ollama 连接"""

    print("="*50)
    print("测试 Ollama 连接...")
    print("="*50)

    try:
        # 创建客户端
        client = Client(host='localhost:11434')

        # 测试连接
        print("\n1. 测试连接...")
        response = client.generate(model='qwen2.5:14b', prompt='你好，请回复"连接成功"')
        print(f"   ✓ 连接成功！")
        print(f"   回复: {response['response'][:100]}...")

        # 测试 JSON 输出
        print("\n2. 测试 JSON 输出...")
        response = client.generate(
            model='qwen2.5:14b',
            prompt='请以 JSON 格式返回：{"status": "ok"}',
            format='json'
        )
        result = response['response']
        print(f"   ✓ JSON 输出成功！")
        print(f"   结果: {result}")

        print("\n" + "="*50)
        print("✓ 所有测试通过！")
        print("="*50)

        return True

    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        print("\n请确保:")
        print("  1. Ollama 已启动: ollama serve")
        print("  2. 模型已下载: ollama pull qwen2.5:14b")
        return False


if __name__ == "__main__":
    test_ollama_connection()
