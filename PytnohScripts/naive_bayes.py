import pandas as pd
import time
import tracemalloc
import psutil
import os
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

RUNS = 5
metrics = {
    "accuracy": [],
    "elapsed_ms": [],
    "cpu_time_ms": [],
    "memory_kb": [],
    "allocated_memory_mb": []
}

df = pd.read_csv("data/mushrooms_cleaned.csv")
X = df.drop("class", axis=1)
y = df["class"]
X_scaled = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

for i in range(RUNS):

    process = psutil.Process(os.getpid())
    cpu_start = process.cpu_times().user
    mem_start = process.memory_info().rss
    tracemalloc.start()
    start_time = time.time()

    clf = GaussianNB()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    elapsed_ms = (time.time() - start_time) * 1000
    cpu_time_ms = (process.cpu_times().user - cpu_start) * 1000
    memory_kb = (process.memory_info().rss - mem_start) / 1024
    allocated_mb = tracemalloc.get_traced_memory()[1] / 1024 / 1024
    tracemalloc.stop()

    metrics["accuracy"].append(accuracy)
    metrics["elapsed_ms"].append(elapsed_ms)
    metrics["cpu_time_ms"].append(cpu_time_ms)
    metrics["memory_kb"].append(memory_kb)
    metrics["allocated_memory_mb"].append(allocated_mb)

# Вывод усреднённых метрик
print("Алгоритм: Naive Bayes (среднее по", RUNS, "запускам)")
print(f"Accuracy: {sum(metrics['accuracy']) / RUNS:.4f}")
print(f"Время выполнения: {sum(metrics['elapsed_ms']) / RUNS:.2f} ms")
print(f"CPU время: {sum(metrics['cpu_time_ms']) / RUNS:.2f} ms")
print(f"Память: {sum(metrics['memory_kb']) / RUNS:.2f} Kb")
print(f"Аллоцировано памяти: {sum(metrics['allocated_memory_mb']) / RUNS:.2f} Mb")
