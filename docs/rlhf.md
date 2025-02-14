
## Policy Model/Actor Model
这个就是我们要训练的目标LLM

## Critic Model/Value Model
预测期望总收益， 需要进行参数更新

## Reward Model


## Reference Model


Policy Model和Critic Model参数是会进行更新的， 而Reward Model和Reference Model参数是冻结的。


# PPO
GPT最原始的RLHF训练时候，使用的策略模型就是PPO



# GRPO
通过组内相对奖励，来训练PPO
1. 移除了critic model
2. 对于一个prompt生成一个group response, 会计算平均reward分数




