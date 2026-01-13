import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def run_task_21():

    print("\n" + "="*80)
    print(" RUNNING TASK 2.1: Crazy Sauce Prediction ")
    print("="*80 + "\n")

    try:
        from task_crazy_sauce import run_experiment
        results, model, features = run_experiment()
        print("\n✓ Task 2.1 completed successfully!")
        return True
    except Exception as e:
        print(f"\n✗ Task 2.1 failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_task_22():

    print("\n" + "="*80)
    print(" RUNNING TASK 2.2: Multi-Sauce Recommendation System ")
    print("="*80 + "\n")

    try:
        from task_22_all_sauce import run_experiment
        recommender, baseline, model_results, baseline_results = run_experiment()
        print("\n✓ Task 2.2 completed successfully!")
        return True
    except Exception as e:
        print(f"\n✗ Task 2.2 failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():

    print("\n" + "="*80)
    print(" ML PROJECT 2025 FALL - RESTAURANT SALES ANALYSIS ")
    print("="*80)

    dataset_path = os.path.join(os.path.dirname(__file__), 'ap_dataset.csv')
    if not os.path.exists(dataset_path):
        print(f"\n✗ ERROR: Dataset not found at {dataset_path}")
        print("Please make sure 'ap_dataset.csv' is in the project root directory.")
        return

    print(f"\n✓ Dataset found: {dataset_path}")

    for dir_name in ['results', 'plots']:
        dir_path = os.path.join(os.path.dirname(__file__), dir_name)
        os.makedirs(dir_path, exist_ok=True)
        print(f"✓ Output directory: {dir_path}")

    results = []

    task_21_success = run_task_21()
    results.append(("Task 2.1", task_21_success))

    print("\n" + "-"*80 + "\n")

    task_22_success = run_task_22()
    results.append(("Task 2.2", task_22_success))

    print("\n" + "="*80)
    print(" SUMMARY ")
    print("="*80)

    for task_name, success in results:
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"{task_name}: {status}")

    all_success = all(success for _, success in results)

    if all_success:
        print("\n✓ All tasks completed successfully!")
        print("\nResults saved to:")
        print("  - results/ directory (CSV files)")
        print("  - plots/ directory (PNG plots)")
    else:
        print("\n⚠ Some tasks failed. Please check the error messages above.")

    print("="*80 + "\n")

if __name__ == "__main__":
    main()
