"""
Reinforcement Learning Environment for Sinhala Call Center Assistant
Integrates feedback-driven parameter optimization with PPO agent training.

Main Components:
- feedback_optimizer: Core feedback analysis and parameter optimization
- ppo_agent: PPO reinforcement learning agent for continuous training
- optimization_manager: Background scheduler, manual control, and integrated training
"""

from RL_enviroment.feedback_optimizer import FeedbackOptimizer, optimize_from_feedback, get_optimized_configs
from RL_enviroment.ppo_agent import PPOAgent, train_ppo_agent, get_ppo_agent, CallCenterEnvironment
from RL_enviroment.optimization_manager import (
    FeedbackOptimizationManager,
    ManualOptimizationTrigger,
    PPOTrainingManager,
    integrate_feedback_optimization,
    train_ppo_with_optimization,
)

__all__ = [
    # Feedback Optimization
    'FeedbackOptimizer',
    'optimize_from_feedback',
    'get_optimized_configs',
    
    # PPO Agent
    'PPOAgent',
    'CallCenterEnvironment',
    'train_ppo_agent',
    'get_ppo_agent',
    
    # Managers
    'FeedbackOptimizationManager',
    'ManualOptimizationTrigger',
    'PPOTrainingManager',
    
    # Integration
    'integrate_feedback_optimization',
    'train_ppo_with_optimization',
]

