"""Main script to run the complete ML pipeline.

This script orchestrates the entire ML pipeline, from data analysis to prediction.
"""

import os
import sys
import logging
import time
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import experiment tracking
from src.experiment_tracking import get_experiment_tracker

def run_pipeline():
    """Run the complete ML pipeline."""
    start_time = time.time()
    logger.info("Starting ML pipeline...")
    
    # Initialize experiment tracking
    tracker = get_experiment_tracker("jigsaw_ml_pipeline")
    with tracker.start_run("full_pipeline"):
        # Log pipeline start
        tracker.log_params({
            "pipeline_start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "pipeline_type": "full"
        })
        
        try:
            # Step 1: Data Analysis
            logger.info("Step 1: Running data analysis...")
            from src.analyze_data import main as analyze_data_main
            analyze_data_main()
            
            # Step 2: Train Baseline Model
            logger.info("Step 2: Training baseline model...")
            from src.train_baseline import main as train_baseline_main
            train_baseline_main()
            
            # Step 3: Train Improved Model
            logger.info("Step 3: Training improved model...")
            from src.train_improved import main as train_improved_main
            train_improved_main()
            
            # Step 4: Fine-tune Model
            logger.info("Step 4: Fine-tuning model...")
            from src.fine_tune_model import main as fine_tune_model_main
            fine_tune_model_main()
            
            # Step 5: Generate Predictions
            logger.info("Step 5: Generating predictions...")
            from src.predict_improved import main as predict_main
            predict_main()
            
            # Log pipeline completion
            end_time = time.time()
            duration = end_time - start_time
            tracker.log_metrics({
                "pipeline_duration_seconds": duration,
                "pipeline_success": 1.0
            })
            
            logger.info(f"ML pipeline completed successfully in {duration:.2f} seconds!")
            return True
            
        except Exception as e:
            # Log pipeline failure
            end_time = time.time()
            duration = end_time - start_time
            tracker.log_metrics({
                "pipeline_duration_seconds": duration,
                "pipeline_success": 0.0
            })
            
            logger.error(f"ML pipeline failed after {duration:.2f} seconds: {str(e)}")
            return False

def main():
    """Main function."""
    success = run_pipeline()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()