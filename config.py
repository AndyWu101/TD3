import argparse


parser = argparse.ArgumentParser()


# 一般設定
parser.add_argument("--algorithm", type = str, default = "TD3", help = "演算法名稱")


# 存檔設定
parser.add_argument("--save_result", action = "store_true", help = "是否存檔")
parser.add_argument("--output_path", type = str, default = "Result", help = "輸出檔案路徑")


# 實驗環境設定
parser.add_argument("--env_name", type = str, default = "HalfCheetah-v5", help = "任務名稱")
parser.add_argument("--device", type = str, default = "cuda:0", help = "實驗使用的設備")
parser.add_argument("--seed", type = int, default = 0, help = "亂數種子")
parser.add_argument("--max_steps", type = int, default = int(1e6), help = "實驗多少步")


# 性能測試設定
parser.add_argument("--test_performance_freq", type = int, default = 5000, help = "每與環境互動多少 steps 要測試一次 actor 性能")
parser.add_argument("--test_n", type = int, default = 20, help = "每次測試 actor 要玩幾局")


# RL設定
parser.add_argument("--replay_buffer_size", type = int, default = int(1e6), help = "replay buffer 的最大空間")
parser.add_argument("--batch_size", type = int, default = 256, help = "random mini-batch size")
parser.add_argument("--start_steps", type = int, default = 25000, help = "最開始使用隨機 action 進行探索")
parser.add_argument("--gamma", type = float, default = 0.99, help = "TD 的 discount")


# TD3設定
parser.add_argument("--actor_learning_rate", type = float, default = 3e-4, help = "actor 的學習率")
parser.add_argument("--critic_learning_rate", type = float, default = 3e-4, help = "critic 的學習率")
parser.add_argument("--exploration_noise", type = float, default = 0.1, help = "與環境互動時對動作加噪，增強 replay buffer 多樣性")
parser.add_argument("--policy_noise", type = float, default = 0.2, help = "對 actor target 做出的 at+1 加噪")
parser.add_argument("--policy_noise_clip", type = float, default = 0.5, help = "policy_noise 的範圍約為(-0.6 , 0.6) (三倍標準差)，再將其 clip 至(-0.5 , 0.5)")
parser.add_argument("--policy_frequency", type = int, default = 2, help = "訓練 critic 次數 與 訓練 actor 次數的比值，同時也是更新 target network 的頻率")
parser.add_argument("--tau", type = float, default = 0.005, help = "以移動平均更新 target 的比例")


args = parser.parse_args()












