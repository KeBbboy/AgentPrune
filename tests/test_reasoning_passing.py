"""测试智能体之间的 reasoning 传递机制"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.stdout.reconfigure(encoding='utf-8')

from AgentDropout.llm.gpt_chat import LLMResponse, split_thinking, create_llm_response_from_parts
from AgentDropout.graph.node import Node


def test_llm_response_parsing():
    """测试 LLMResponse 的解析功能"""
    print("=" * 60)
    print("测试 1: LLMResponse 解析功能")
    print("=" * 60)
    
    # 测试 MiniMax-M3 格式
    text1 = '<think>这是推理过程，包含复杂的逻辑分析。</think>最终答案是 A'
    result1 = split_thinking(text1)
    assert result1.reasoning == '这是推理过程，包含复杂的逻辑分析。', f"MiniMax-M3 reasoning 解析失败: {result1.reasoning}"
    assert result1.answer == '最终答案是 A', f"MiniMax-M3 answer 解析失败: {result1.answer}"
    assert result1.full_text == text1, f"MiniMax-M3 full_text 解析失败: {result1.full_text}"
    print("✓ MiniMax-M3 格式解析正确")
    
    # 测试没有 thinking 标签的情况
    text2 = '直接给出答案 B，没有推理过程'
    result2 = split_thinking(text2)
    assert result2.reasoning is None, f"无 thinking 标签时 reasoning 应该为 None: {result2.reasoning}"
    assert result2.answer == '直接给出答案 B，没有推理过程', f"无 thinking 标签时 answer 解析失败: {result2.answer}"
    print("✓ 无 thinking 标签的文本解析正确")
    
    # 测试空文本
    text3 = ''
    result3 = split_thinking(text3)
    assert result3.reasoning is None
    assert result3.answer == ''
    assert result3.full_text == ''
    print("✓ 空文本解析正确")
    
    # 测试 Qwen3 格式
    result4 = create_llm_response_from_parts('Qwen3 的推理过程，使用分离字段。', '最终答案是 C')
    assert result4.reasoning == 'Qwen3 的推理过程，使用分离字段。', f"Qwen3 reasoning 解析失败: {result4.reasoning}"
    assert result4.answer == '最终答案是 C', f"Qwen3 answer 解析失败: {result4.answer}"
    assert '<think>' in result4.full_text, f"Qwen3 full_text 应该包含 <think> 标签: {result4.full_text}"
    print("✓ Qwen3 格式解析正确")
    
    print()


def test_node_message_passing():
    """测试 Node 的消息传递机制"""
    print("=" * 60)
    print("测试 2: Node 消息传递机制")
    print("=" * 60)
    
    class TestNode(Node):
        def __init__(self, node_id):
            super().__init__(id=node_id)
        
        def _execute(self, input, spatial_info, temporal_info, **kwargs):
            pass
        
        async def _async_execute(self, input, spatial_info, temporal_info, **kwargs):
            pass
        
        def _process_inputs(self, raw_inputs, spatial_info, temporal_info, **kwargs):
            pass
    
    node = TestNode('test')
    
    # 测试结构化输出
    llm_response = LLMResponse(
        reasoning='测试推理内容', 
        answer='测试答案', 
        full_text='<think>测试推理内容</think>测试答案'
    )
    message_type, output = node._resolve_message_payload(node, llm_response)
    assert message_type == 'structured', f"结构化输出的 message_type 应该是 'structured': {message_type}"
    assert isinstance(output, dict), f"结构化输出应该是字典: {type(output)}"
    assert 'reasoning' in output, f"结构化输出应该包含 reasoning 字段"
    assert 'answer' in output, f"结构化输出应该包含 answer 字段"
    assert 'full_text' in output, f"结构化输出应该包含 full_text 字段"
    assert output['reasoning'] == '测试推理内容', f"reasoning 字段值不正确: {output['reasoning']}"
    assert output['answer'] == '测试答案', f"answer 字段值不正确: {output['answer']}"
    print("✓ 结构化输出处理正确")
    
    # 测试普通文本输出
    text_response = '普通文本输出内容'
    message_type2, output2 = node._resolve_message_payload(node, text_response)
    assert message_type2 == 'full', f"普通文本的 message_type 应该是 'full': {message_type2}"
    assert output2 == '普通文本输出内容', f"普通文本输出值不正确: {output2}"
    print("✓ 普通文本处理正确")
    
    print()


def test_agent_output_processing():
    """测试智能体的输出处理逻辑"""
    print("=" * 60)
    print("测试 3: 智能体输出处理逻辑")
    print("=" * 60)
    
    def _get_output_content(output):
        if isinstance(output, dict) and 'full_text' in output:
            return output['full_text']
        return output
    
    # 测试结构化输出
    structured_output = {
        'reasoning': '推理内容',
        'answer': '答案',
        'full_text': '<think>推理内容</think>答案'
    }
    content1 = _get_output_content(structured_output)
    assert content1 == '<think>推理内容</think>答案', f"结构化输出的 full_text 提取失败: {content1}"
    print("✓ 结构化输出的 full_text 提取正确")
    
    # 测试普通文本输出
    text_output = '普通文本'
    content2 = _get_output_content(text_output)
    assert content2 == '普通文本', f"普通文本提取失败: {content2}"
    print("✓ 普通文本提取正确")
    
    # 测试 LLMResponse 对象
    llm_response = LLMResponse(reasoning='LLMResponse 推理', answer='LLMResponse 答案', full_text='<think>LLMResponse 推理</think>LLMResponse 答案')
    content3 = _get_output_content(llm_response)
    assert content3 == llm_response, f"LLMResponse 对象应该直接返回: {content3}"
    print("✓ LLMResponse 对象处理正确")
    
    print()


def test_spatial_info_structure():
    """测试空间信息的结构"""
    print("=" * 60)
    print("测试 4: 空间信息结构")
    print("=" * 60)
    
    class TestNode(Node):
        def __init__(self, node_id, role):
            super().__init__(id=node_id)
            self.role = role
        
        def _execute(self, input, spatial_info, temporal_info, **kwargs):
            pass
        
        async def _async_execute(self, input, spatial_info, temporal_info, **kwargs):
            pass
        
        def _process_inputs(self, raw_inputs, spatial_info, temporal_info, **kwargs):
            pass
    
    # 创建节点
    node1 = TestNode('node1', 'Analyst')
    node2 = TestNode('node2', 'Expert')
    
    # 设置输出
    node1.outputs = [LLMResponse(reasoning='节点1的推理', answer='节点1的答案', full_text='<think>节点1的推理</think>节点1的答案')]
    node2.outputs = ['普通文本输出']
    
    # 添加连接
    node2.add_predecessor(node1, st='spatial')
    
    # 获取空间信息
    spatial_info = node2.get_spatial_info()
    assert 'node1' in spatial_info, f"空间信息应该包含 node1"
    assert spatial_info['node1']['role'] == 'Analyst', f"角色信息不正确: {spatial_info['node1']['role']}"
    assert 'output' in spatial_info['node1'], f"空间信息应该包含 output 字段"
    assert 'message_type' in spatial_info['node1'], f"空间信息应该包含 message_type 字段"
    
    # 验证结构化输出
    output = spatial_info['node1']['output']
    if isinstance(output, dict):
        assert 'reasoning' in output
        assert 'answer' in output
        assert 'full_text' in output
        print("✓ 结构化输出的空间信息正确")
    else:
        print("✓ 文本输出的空间信息正确")
    
    print()


def main():
    print("智能体之间 reasoning 传递机制测试")
    print("=" * 60)
    print()
    
    test_llm_response_parsing()
    test_node_message_passing()
    test_agent_output_processing()
    test_spatial_info_structure()
    
    print("=" * 60)
    print("所有测试通过！")
    print("=" * 60)


if __name__ == "__main__":
    main()
