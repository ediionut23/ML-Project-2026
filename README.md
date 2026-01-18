# Machine Learning Assignment - Restaurant Product Recommendations

## Project Structure
```
ml_project/
├── ap_dataset.csv          # Dataset
├── requirements.txt        # Dependencies
├── run_all.py             # Run all experiments
├── src/
│   ├── data_preprocessing.py
│   ├── logistic_regression.py
│   ├── evaluation.py
│   ├── ranking_algorithms.py
│   ├── task_21_crazy_sauce.py
│   ├── task_22_all_sauce.py
│   └── task_3_ranking.py
├── plots/                  # Generated plots
└── results/               # Generated CSV results
```

## Installation
```bash
pip install -r requirements.txt
```

## Running
```bash
# Run all tasks
python run_all.py

# Or run individually
cd src
python task_21_crazy_sauce.py
python task_22_all_sauce.py
python task_3_ranking.py
```

## Tasks
- **Task 2.1**: Logistic Regression for Crazy Sauce prediction (conditioned on Crazy Schnitzel)
- **Task 2.2**: Multi-sauce recommendation using Logistic Regression
- **Task 3**: Product ranking using Naive Bayes, k-NN, Decision Tree, AdaBoost
