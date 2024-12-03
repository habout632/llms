

# Self Play RL Agent
自主学习模型，基于强化学习，通过自我学习，学习到最优策略。

现在面临的主要问题是, 以测试用例生成举例
1. 在每一步当中，都会有些问题 没法顺利的进入到 下一步

生成的测试用例当中 有些已经超出了api文档的范围，有些必须过滤掉

根据feature生成step definiton常常有些报错， 怎么能自我修复错误 重新生成


## Papers
[Quiet-STaR: Language Models Can Teach Themselves to Think Before Speaking](https://arxiv.org/abs/2403.09629)

[Mutual Reasoning Makes Smaller LLMs Stronger Problem-Solvers](https://arxiv.org/abs/2408.06195)

## Frameworks

FLAML (Fast and Lightweight AutoML)：虽然主要focus在自动机器学习，但它也提供了一些LLM相关的功能，特别是在优化提示和模型选择方面。

AutoGPT：
这是最著名的自主AI代理框架之一。AutoGPT能够自主地设定目标、规划步骤、执行任务，并根据结果进行自我评估和调整。它使用GPT-4作为其核心引擎，能够处理复杂的、多步骤的任务。

BabyAGI：
这是一个简化版的自主AI代理系统。它能够创建任务列表，确定优先级，并逐个执行任务。BabyAGI的设计理念是简单且可扩展的，使得研究人员和开发者可以轻松地在其基础上构建更复杂的系统。

AgentGPT：
类似于AutoGPT，AgentGPT是一个基于网络的平台，允许用户创建和部署自主AI代理。它提供了一个用户友好的界面，使得非技术用户也能轻松创建自己的AI代理。

CAMEL (Communicative Agents for "Mind" Exploration of Large Scale Language Model Society)：
这是一个框架，专注于创建能够相互交流的AI代理。CAMEL允许多个AI代理协作完成任务，模拟复杂的社会互动。

LangChain Agents：
虽然LangChain本身不是专门的自主学习框架，但它的Agents模块提供了创建自主AI代理的强大工具。这些代理可以使用工具、进行推理，并自主地完成复杂任务。