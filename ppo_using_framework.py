from transformers import AutoTokenizer, AutoModelForCausalLM  
from datasets import load_dataset  
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead  

import gym  
from stable_baselines3 import PPO  
from stable_baselines3.common.env_util import make_vec_env  
from stable_baselines3.common.evaluation import evaluate_policy 





class FrameWorkPPO:
    
    
    def __init__(self, model_name_or_path, ppo_config=None):
        
        self.ppo_config = ppo_config if ppo_config else PPOConfig(
            batch_size=32,  
            learning_rate=1.41e-5,  
            mini_batch_size=4,  
            init_kl_coef=0.2,  # KL散度惩罚系数  
            target_kl=6,       # 目标KL散度阈值  
            ppo_epochs=4,      # PPO优化轮次  
        )
        
        self.model, self.tokenizer = self._initialize_model_and_tokenizer(model_name_or_path)
    
        self.dataset = load_dataset("imdb", split="train[:1%]") 
        
        
        self.ppo_trainer = PPOTrainer(  
            model=self.model,  
            config=self.ppo_config,  
            tokenizer=self.tokenizer,  
            dataset=self.dataset,  
        )  
        
        
    
    def reward_model(self, texts):
        # 这里应替换为实际的奖励模型计算逻辑  
        # 可以是分类模型输出、人工规则或人类反馈  
        return [1.0 if "positive" in text else -1.0 for text in texts]  
    
    
    
    
    def train_using_hf(self):
        for epoch in range(3):  
            for batch in ppo_trainer.dataloader:  
                # 生成文本  
                query_tensors = tokenizer(batch["text"], return_tensors="pt", padding=True).input_ids  
                
                # 模型生成响应  
                response_tensors = ppo_trainer.generate(  
                    query_tensors,  
                    max_new_tokens=50,  
                    temperature=0.7,  
                    top_k=50  
                )  
                
                # 计算奖励  
                responses = tokenizer.batch_decode(response_tensors)  
                rewards = reward_model(responses)  
                
                # PPO优化步骤  
                stats = ppo_trainer.step(query_tensors, response_tensors, rewards)  
                print(f"Epoch {epoch} | Reward: {sum(rewards)/len(rewards):.2f}")  
                
                
    def train_using_stable_baseline(self):
        # 创建并行环境  
        env = make_vec_env("CartPole-v1", n_envs=4)  

        # 初始化PPO模型  
        model = PPO(  
            "MlpPolicy",   
            env,  
            learning_rate=3e-4,  
            n_steps=1024,       # 每次更新前运行的步数  
            batch_size=64,      # 经验回放缓冲区大小  
            n_epochs=10,        # 优化时的epoch数  
            gamma=0.99,         # 折扣因子  
            gae_lambda=0.95,    # GAE参数  
            clip_range=0.2,     # 裁剪范围（核心PPO参数）  
            verbose=1  
        )  

        # 训练模型  
        model.learn(total_timesteps=100000)  

        # 评估模型  
        mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=10)  
        print(f"Mean reward: {mean_reward:.2f}")  

        # 保存模型  
        model.save("ppo_cartpole") 