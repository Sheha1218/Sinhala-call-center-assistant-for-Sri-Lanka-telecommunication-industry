
import time
import threading
from datetime import datetime
try:
    import schedule
except ImportError:
    schedule = None
from RL_enviroment.feedback_optimizer import FeedbackOptimizer
from RL_enviroment.ppo_agent import PPOAgent, get_ppo_agent


class FeedbackOptimizationManager:
   
    
    def __init__(self, check_interval_minutes=60):
        
        self.optimizer = FeedbackOptimizer()
        self.check_interval = check_interval_minutes
        self.is_running = False
        self.thread = None
    
    def check_and_optimize(self):
        """Check feedback and trigger optimization if threshold is met."""
        print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Checking feedback for optimization...")
        result = self.optimizer.optimize()
        
        if result['status'] == 'success':
            print(f"Successfully optimized model parameters based on {result['feedback_count']} feedbacks")
            print(f" Average feedback score: {result['avg_feedback_value']:.2f}/10")
            return True
        else:
            print(f"ℹ Optimization check: {result['status']}")
            if 'current_feedbacks' in result:
                print(f" Current feedbacks: {result['current_feedbacks']}/{result['required_feedbacks']}")
        
        return False
    
    def start_background_scheduler(self):
        if not schedule:
            return False
        
        if self.is_running:
            print("⚠ Scheduler is already running")
            return False
        
        self.is_running = True
        
        
        schedule.every(self.check_interval).minutes.do(self.check_and_optimize)
        
        
        self.thread = threading.Thread(target=self._run_scheduler, daemon=True)
        self.thread.name = "FeedbackOptimizer"
        self.thread.start()
        
        print(f"✓ Background feedback optimizer started (checks every {self.check_interval} minutes)")
        return True
    
    def _run_scheduler(self):
        """Internal scheduler loop (runs in thread)."""
        while self.is_running:
            try:
                if schedule:
                    schedule.run_pending()
                time.sleep(60)  # Check every minute if there's a task to run
            except Exception as e:
                print(f"Error in feedback optimizer: {str(e)}")
                time.sleep(60)
    
    def stop_background_scheduler(self):
        """Stop the background scheduler."""
        self.is_running = False
        if self.thread:
            self.thread.join(timeout=5)
        print("✓ Background feedback optimizer stopped")
    
    def run_once(self):
        """Run optimization check once (blocking)."""
        return self.check_and_optimize()


class ManualOptimizationTrigger:
    def __init__(self):
        self.optimizer = FeedbackOptimizer()
    
    def trigger_optimization(self):
        print("\n" + "="*70)
        print("MANUAL FEEDBACK OPTIMIZATION TRIGGER")
        print("="*70)
        return self.optimizer.optimize()
    
    def get_feedback_report(self):
        print("\n" + "="*70)
        print("FEEDBACK DATA REPORT")
        print("="*70)
        
        feedback_df = self.optimizer.fetch_feedback_data()
        
        if feedback_df.empty:
            print("No feedback data available")
            return None
        
        metrics = self.optimizer.calculate_feedback_metrics(feedback_df)
        
        print(f"\nTotal Feedbacks: {metrics['total_feedbacks']}")
        print(f"Average Rating: {metrics['avg_feedback_value']:.2f}/10")
        print(f"Median Rating: {metrics['median_feedback_value']:.2f}/10")
        print(f"Std Deviation: {metrics['std_feedback_value']:.2f}")
        print(f"Range: {metrics['min_feedback_value']} - {metrics['max_feedback_value']}")
        print(f"\nPositive Feedbacks (≥7): {metrics['positive_feedback_count']} ({metrics['positive_feedback_count']/metrics['total_feedbacks']*100:.1f}%)")
        print(f"Negative Feedbacks (<5): {metrics['negative_feedback_count']} ({metrics['negative_feedback_count']/metrics['total_feedbacks']*100:.1f}%)")
        
        # Show threshold status
        threshold_status = "✓ READY FOR OPTIMIZATION" if metrics['total_feedbacks'] >= self.optimizer.FEEDBACK_THRESHOLD else "✗ NOT YET"
        print(f"\nOptimization Threshold Status: {threshold_status}")
        print(f"Progress: {metrics['total_feedbacks']}/{self.optimizer.FEEDBACK_THRESHOLD}")
        
        print("\n" + "="*70 + "\n")
        
        return metrics
    
    def get_current_config(self):
        """Display the current active configuration."""
        print("\n" + "="*70)
        print("CURRENT ACTIVE CONFIGURATION")
        print("="*70)
        
        config = self.optimizer.load_current_config()
        
        print("\nPPO Configuration:")
        for key, value in config['ppo_config'].items():
            print(f"  {key}: {value}")
        
        print("\nLoRA Configuration:")
        for key, value in config['lora_config'].items():
            print(f"  {key}: {value}")
        
        print(f"\nLast Updated: {config.get('last_updated', 'Never')}")
        print(f"Optimization History: {len(config.get('optimization_history', []))} records")
        
        if config.get('optimization_history'):
            latest = config['optimization_history'][-1]
            print(f"\nLatest Optimization:")
            print(f"  Timestamp: {latest['timestamp']}")
            print(f"  Feedback Count: {latest['feedback_count']}")
            print(f"  Avg Feedback: {latest['avg_feedback_value']:.2f}/10")
            print(f"  Positive Ratio: {latest['positive_ratio']:.1%}")
        
        print("\n" + "="*70 + "\n")


class PPOTrainingManager:
    """
    Manages PPO agent training with automated feedback collection.
    Combines PPO reinforcement learning with parameter optimization.
    """
    
    def __init__(self):
        self.ppo_agent = PPOAgent(state_dim=768, action_dim=256)
        self.feedback_optimizer = FeedbackOptimizer()
        self.training_history = []
    
    def train_and_optimize(self, num_episodes=100):
        """
        Comprehensive training pipeline:
        1. Train PPO agent on feedback
        2. Optimize parameters based on aggregate feedback
        
        Args:
            num_episodes: Number of feedback episodes to train on
            
        Returns:
            dict: Training and optimization results
        """
        print("\n" + "="*80)
        print("PPO TRAINING + PARAMETER OPTIMIZATION PIPELINE")
        print("="*80)
        
        # Phase 1: PPO Training
        print("\n[PHASE 1/2] PPO REINFORCEMENT LEARNING TRAINING")
        print("-" * 80)
        training_result = self.ppo_agent.train_on_feedback(num_episodes=num_episodes)
        
        if training_result['status'] != 'success':
            print("⚠ Training failed, skipping optimization phase")
            return training_result
        
        # Phase 2: Parameter Optimization
        print("\n[PHASE 2/2] PARAMETER OPTIMIZATION")
        print("-" * 80)
        optimization_result = self.feedback_optimizer.optimize()
        
        # Combine results
        results = {
            'status': 'complete',
            'training': training_result,
            'optimization': optimization_result,
            'timestamp': datetime.now().isoformat(),
        }
        
        # Save checkpoint
        self.ppo_agent.save_checkpoint()
        
        print("\n" + "="*80)
        print("✓ PIPELINE COMPLETE")
        print("="*80)
        print(f"Training Episodes: {training_result['episodes']}")
        print(f"Average Training Reward: {training_result['avg_reward']:.4f}")
        if optimization_result['status'] == 'success':
            print(f"Parameters Optimized: ✓")
            print(f"Feedback Count: {optimization_result['feedback_count']}")
        else:
            print(f"Parameters Optimized: ✗ ({optimization_result['status']})")
        print("="*80 + "\n")
        
        self.training_history.append(results)
        return results
    
    def get_training_stats(self):
        """Get training statistics."""
        stats = self.ppo_agent.get_training_stats()
        return stats
    
    def get_training_history(self):
        """Get training history."""
        return self.training_history


# Integration Helper
def integrate_feedback_optimization(app=None, use_background=True, check_interval=60):
    manager = FeedbackOptimizationManager(check_interval_minutes=check_interval)
    
    if use_background:
        manager.start_background_scheduler()
    
    # If FastAPI app is provided, optionally add shutdown hook
    if app:
        @app.on_event("shutdown")
        async def shutdown_optimizer():
            manager.stop_background_scheduler()
    
    return manager


# Convenience function for PPO training
def train_ppo_with_optimization(num_episodes=100):
    """
    Train PPO agent and optimize parameters in one integrated pipeline.
    
    Args:
        num_episodes: Number of feedback episodes to train on
        
    Returns:
        dict: Combined results from training and optimization
    """
    manager = PPOTrainingManager()
    return manager.train_and_optimize(num_episodes=num_episodes)


