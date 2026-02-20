import os
import json
import pandas as pd
from datetime import datetime
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import numpy as np
from pathlib import Path

load_dotenv()

class FeedbackOptimizer:
   
    
    
    DEFAULT_PPO_CONFIG = {
        'learning_rate': 5e-6,
        'batch_size': 5,
        'mini_batch_size': 1,
        'num_train_epochs': 4,
        'gradient_accumulation_steps': 1,
    }
    
    DEFAULT_LORA_CONFIG = {
        'r': 8,
        'lora_alpha': 16,
        'target_modules': ['q_proj', 'v_proj'],
        'lora_dropout': 0.05,
        'bias': 'none',
        'task_type': 'CAUSAL_LM'
    }
    
    FEEDBACK_THRESHOLD = 100  
    FEEDBACK_HISTORY_FILE = './RL_enviroment/feedback_history.json'
    OPTIMIZED_CONFIG_FILE = './RL_enviroment/optimized_config.json'
    
    def __init__(self):
        
        self.db_url = os.getenv('db_url', 'postgresql+psycopg2://postgres:root@localhost:5432/call_agent')
        self.engine = create_engine(self.db_url)
        self.feedback_table = 'customer_feedback'
        
        # Initialize config files if they don't exist
        self._initialize_config_files()
    
    def _initialize_config_files(self):
        """Initialize configuration files if they don't exist."""
        if not os.path.exists(self.OPTIMIZED_CONFIG_FILE):
            config = {
                'ppo_config': self.DEFAULT_PPO_CONFIG,
                'lora_config': self.DEFAULT_LORA_CONFIG,
                'last_updated': None,
                'optimization_history': []
            }
            with open(self.OPTIMIZED_CONFIG_FILE, 'w') as f:
                json.dump(config, f, indent=4)
    
    def fetch_feedback_data(self):
        query = f"""
            SELECT feedback_id, customer_nic, connection_number, 
                   feedback_value, feedback_message, created_at 
            FROM {self.feedback_table}
            ORDER BY created_at DESC;
        """
        try:
            with self.engine.connect() as conn:
                result = pd.read_sql_query(text(query), conn)
            return result
        except Exception as e:
            print(f"Error fetching feedback data: {str(e)}")
            return pd.DataFrame()
    
    def calculate_feedback_metrics(self, feedback_df):
        if feedback_df.empty:
            return None
        
        metrics = {
            'total_feedbacks': len(feedback_df),
            'avg_feedback_value': feedback_df['feedback_value'].mean() if 'feedback_value' in feedback_df.columns else 0,
            'median_feedback_value': feedback_df['feedback_value'].median() if 'feedback_value' in feedback_df.columns else 0,
            'std_feedback_value': feedback_df['feedback_value'].std() if 'feedback_value' in feedback_df.columns else 0,
            'min_feedback_value': feedback_df['feedback_value'].min() if 'feedback_value' in feedback_df.columns else 0,
            'max_feedback_value': feedback_df['feedback_value'].max() if 'feedback_value' in feedback_df.columns else 0,
            'positive_feedback_count': len(feedback_df[feedback_df['feedback_value'] >= 7]) if 'feedback_value' in feedback_df.columns else 0,
            'negative_feedback_count': len(feedback_df[feedback_df['feedback_value'] < 5]) if 'feedback_value' in feedback_df.columns else 0,
        }
        
        return metrics
    
    def get_next_optimization_threshold(self):
        
        current_config = self.load_current_config()
        optimization_count = len(current_config.get('optimization_history', []))
        next_threshold = (optimization_count + 1) * self.FEEDBACK_THRESHOLD
        return next_threshold
    
    def should_optimize(self, metrics):
      
        if not metrics:
            return False
        
        next_threshold = self.get_next_optimization_threshold()
        return metrics['total_feedbacks'] >= next_threshold
    
    def calculate_optimization_adjustments(self, metrics):
      
        adjustments = {
            'ppo_multiplier': 1.0,  # Multiplier for learning rate
            'lora_multiplier': 1.0,  # Multiplier for LoRA parameters
            'rationale': []
        }
        
        avg_feedback = metrics['avg_feedback_value']
        positive_ratio = metrics['positive_feedback_count'] / metrics['total_feedbacks']
        
        # Feedback-based adjustments
        if avg_feedback < 4:
            # Poor performance - increase learning rate for faster adaptation
            adjustments['ppo_multiplier'] = 1.5
            adjustments['lora_multiplier'] = 1.3
            adjustments['rationale'].append(f"Low average feedback ({avg_feedback:.2f}) - increasing learning capacity")
        
        elif avg_feedback < 6:
            # Average performance - slight increase in learning
            adjustments['ppo_multiplier'] = 1.2
            adjustments['lora_multiplier'] = 1.1
            adjustments['rationale'].append(f"Average feedback ({avg_feedback:.2f}) - modest improvement needed")
        
        elif avg_feedback >= 7:
            # Good performance - slight optimization to fine-tune
            adjustments['ppo_multiplier'] = 0.9
            adjustments['lora_multiplier'] = 0.95
            adjustments['rationale'].append(f"Good average feedback ({avg_feedback:.2f}) - fine-tuning mode")
        
        # Consistency-based adjustments
        if metrics['std_feedback_value'] > 2.5:
            adjustments['rationale'].append(f"High variability (std={metrics['std_feedback_value']:.2f}) - increasing batch size for stability")
        
        return adjustments
    
    def optimize_ppo_config(self, base_config, adjustments):
       
        optimized = base_config.copy()
        
        # Adjust learning rate
        optimized['learning_rate'] = float(base_config['learning_rate'] * adjustments['ppo_multiplier'])
        
        # Adjust batch size based on feedback consistency
        if adjustments['ppo_multiplier'] > 1.2:
            optimized['batch_size'] = max(4, int(base_config['batch_size'] * adjustments['ppo_multiplier']))
            optimized['mini_batch_size'] = max(1, int(base_config['mini_batch_size'] * 1.2))
        else:
            optimized['batch_size'] = base_config['batch_size']
            optimized['mini_batch_size'] = base_config['mini_batch_size']
        
        # Clamp values to reasonable ranges
        optimized['learning_rate'] = max(1e-7, min(1e-4, optimized['learning_rate']))
        optimized['batch_size'] = max(1, min(128, optimized['batch_size']))
        
        return optimized
    
    def optimize_lora_config(self, base_config, adjustments):
        optimized = base_config.copy()
        
        # Adjust LoRA rank (r) - affects model capacity
        optimized['r'] = max(4, min(64, int(base_config['r'] * adjustments['lora_multiplier'])))
        
        # Adjust LoRA alpha (alpha) - scaling factor
        optimized['lora_alpha'] = max(8, min(128, int(base_config['lora_alpha'] * adjustments['lora_multiplier'])))
        
        # Adjust dropout for regularization
        if adjustments['lora_multiplier'] > 1.2:
            optimized['lora_dropout'] = min(0.1, base_config['lora_dropout'] * 1.2)
        elif adjustments['lora_multiplier'] < 0.95:
            optimized['lora_dropout'] = max(0.01, base_config['lora_dropout'] * 0.8)
        
        return optimized
    
    def load_current_config(self):
        """Load the current optimized configuration from file."""
        try:
            with open(self.OPTIMIZED_CONFIG_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading config: {str(e)}")
            return {
                'ppo_config': self.DEFAULT_PPO_CONFIG.copy(),
                'lora_config': self.DEFAULT_LORA_CONFIG.copy(),
                'last_updated': None,
                'optimization_history': []
            }
    
    def save_optimized_config(self, optimized_ppo, optimized_lora, metrics, adjustments):
      
        current_config = self.load_current_config()
        
        # Create optimization record
        optimization_record = {
            'timestamp': datetime.now().isoformat(),
            'feedback_count': metrics['total_feedbacks'],
            'avg_feedback_value': float(metrics['avg_feedback_value']),
            'positive_ratio': float(metrics['positive_feedback_count'] / metrics['total_feedbacks']),
            'ppo_config': optimized_ppo,
            'lora_config': optimized_lora,
            'adjustments_applied': adjustments['rationale']
        }
        
        # Update config file
        current_config['ppo_config'] = optimized_ppo
        current_config['lora_config'] = optimized_lora
        current_config['last_updated'] = datetime.now().isoformat()
        current_config['optimization_history'].append(optimization_record)
        
        # Keep only last 50 optimization records to avoid file bloat
        if len(current_config['optimization_history']) > 50:
            current_config['optimization_history'] = current_config['optimization_history'][-50:]
        
        try:
            os.makedirs(os.path.dirname(self.OPTIMIZED_CONFIG_FILE), exist_ok=True)
            with open(self.OPTIMIZED_CONFIG_FILE, 'w') as f:
                json.dump(current_config, f, indent=4)
            print(f"✓ Configuration saved successfully")
        except Exception as e:
            print(f"Error saving config: {str(e)}")
    
    def optimize(self):
        feedback_df = self.fetch_feedback_data()
        
        if feedback_df.empty:
    
            return {'status': 'no_data', 'message': 'No feedback records available'}
        
        
        metrics = self.calculate_feedback_metrics(feedback_df)
        
        if not self.should_optimize(metrics):
            next_threshold = self.get_next_optimization_threshold()
            return {
                'status': 'threshold_not_met',
                'current_feedbacks': metrics['total_feedbacks'],
                'required_feedbacks': next_threshold
            }
        
        
        
        
        print(f"\n[4/5] Calculating parameter adjustments...")
        adjustments = self.calculate_optimization_adjustments(metrics)
        for rationale in adjustments['rationale']:
            print(f"     • {rationale}")
        
        
        current_config = self.load_current_config()
        base_ppo = current_config['ppo_config']
        base_lora = current_config['lora_config']
        
        
        optimized_ppo = self.optimize_ppo_config(base_ppo, adjustments)
        optimized_lora = self.optimize_lora_config(base_lora, adjustments)
        
       
        
        self.save_optimized_config(optimized_ppo, optimized_lora, metrics, adjustments)
        
      
        return {
            'status': 'success',
            'feedback_count': metrics['total_feedbacks'],
            'avg_feedback_value': metrics['avg_feedback_value'],
            'ppo_config': optimized_ppo,
            'lora_config': optimized_lora,
            'adjustments': adjustments['rationale']
        }
    
    def get_current_config(self):
        config = self.load_current_config()
        return config['ppo_config'], config['lora_config']


# Standalone functions for easy integration
def optimize_from_feedback():
    optimizer = FeedbackOptimizer()
    return optimizer.optimize()


def get_optimized_configs():
    optimizer = FeedbackOptimizer()
    return optimizer.get_current_config()


