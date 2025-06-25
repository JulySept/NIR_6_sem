import pandas as pd
from sklearn.model_selection import train_test_split
from experiment.experiment_runner import ExperimentRunner
#from algorithms.prior_runner import PriorRunner
from algorithms.naive_bayes_runner import NaiveBayesRunner
#from algorithms.linear_svm_runner import LinearSvmRunner
# from algorithms.lds_svm_runner import LdSvmRunner

def main():
    data_path = "data/mushrooms_cleaned.csv"
    df = pd.read_csv(data_path)

    X = df.drop(columns=["class"])
    y = df["class"]

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    algorithms = [
        #PriorRunner(),
        NaiveBayesRunner(),
        #LinearSvmRunner(),
        # LdSvmRunner(),  # опционально
    ]

    runs_count = 5
    runner = ExperimentRunner(x_train, x_test, y_train, y_test)

    for algo in algorithms:
        print(f"Запускаем алгоритм: {algo.name}")

        result = runner.run_experiment(algo, runs_count)

        print(f"Результаты эксперимента для {algo.name}:")
        print(f"  Время выполнения: {result['elapsed_ms']:.2f} ms")
        print(f"  CPU время: {result['cpu_time_ms']:.2f} ms")
        print(f"  Память: {result['memory_kb']:.2f} KB")
        print(f"  Аллоцировано памяти: {result['allocated_memory_mb']:.2f} MB")

        if "metrics" in result:
            print("  Метрики модели:")
            for name, value in result["metrics"].items():
                print(f"    {name}: {value:.4f}")
        print()

if __name__ == "__main__":
    main()
