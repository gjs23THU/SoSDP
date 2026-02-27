"""Simple test script for SoSEnv"""

import torch
import numpy as np
from SoS_env import SoSEnv

def test_env_basic():
    """Test basic environment functionality"""
    
    print("=" * 60)
    print("Testing SoSEnv Basic Functionality")
    print("=" * 60)
    
    # Create environment
    print("\n1. Creating SoSEnv...")
    env = SoSEnv(num_systems=25, num_samples=100, num_tasks=5, seed=42)
    print(f"   ✓ Environment created")
    print(f"   - Action space: {env.action_space}")
    print(f"   - Num nodes: {env.num_nodes}")
    print(f"   - Max steps: {env.max_steps}")
    
    # Reset environment
    print("\n2. Resetting environment...")
    obs, info = env.reset()
    print(f"   ✓ Reset successful")
    print(f"   - Static shape: {obs['static'].shape}")
    print(f"   - Dynamic shape: {obs['dynamic'].shape}")
    print(f"   - Mask shape: {obs['mask'].shape}")
    print(f"   - T shape: {obs['t'].shape}")
    print(f"   - Step count: {obs['step_count']}")
    print(f"   - Mask sum (valid actions): {info['mask_sum']}")
    print(f"   - T sum (total requirements): {info['t_sum']:.2f}")
    
    # Run episode
    print("\n3. Running episode with random actions...")
    episode_reward = 0
    done = False
    steps = 0
    
    while not done and steps < 10:
        # Random valid action
        valid_actions = torch.where(obs['mask'] > 0)[0].tolist()
        if not valid_actions:
            break
        
        action = np.random.choice(valid_actions)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        steps += 1
        
        print(f"\n   Step {steps}:")
        print(f"   - Action: {action}")
        print(f"   - T sum: {info['t_sum']:.2f}")
        print(f"   - Mask sum: {info['mask_sum']}")
        print(f"   - Task counts: {info['task_counts']}")
        print(f"   - Terminated: {terminated}, Truncated: {truncated}")
    
    # Compute episode reward
    print("\n4. Computing final episode reward...")
    final_reward = env._compute_episode_reward()
    print(f"   ✓ Final reward: {final_reward:.4f}")
    print(f"   - Steps taken: {steps}")
    print(f"   - Selected nodes: {len(info['selected_nodes'])}")
    
    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)

if __name__ == '__main__':
    test_env_basic()
