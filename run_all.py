#!/usr/bin/env python3
"""
ML Project 2026 - Run All Experiments
=====================================
This script runs all three tasks sequentially:
- Task 2.1: Logistic Regression for Crazy Sauce prediction
- Task 2.2: Multi-sauce recommendation system
- Task 3: Product ranking for upsell recommendations
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def main():
    print("=" * 80)
    print("ML PROJECT 2026 - RESTAURANT PRODUCT RECOMMENDATIONS")
    print("=" * 80)
    
    # Task 2.1
    print("\n" + "=" * 80)
    print("TASK 2.1: Logistic Regression - Crazy Sauce | Crazy Schnitzel")
    print("=" * 80)
    from task_21_crazy_sauce import run_experiment as run_task21
    run_task21()
    
    # Task 2.2
    print("\n" + "=" * 80)
    print("TASK 2.2: Multi-Sauce Recommendation System")
    print("=" * 80)
    from task_22_all_sauce import run_experiment as run_task22
    run_task22()
    
    # Task 3
    print("\n" + "=" * 80)
    print("TASK 3: Product Ranking for Upsell")
    print("=" * 80)
    from task_3_ranking import run_comprehensive_ranking_experiment as run_task3
    run_task3()
    
    print("\n" + "=" * 80)
    print("ALL EXPERIMENTS COMPLETED!")
    print("=" * 80)
    print("\nGenerated outputs:")
    print("  - plots/          : All generated plots")
    print("  - results/        : CSV result files")
    print("=" * 80)

if __name__ == "__main__":
    main()
