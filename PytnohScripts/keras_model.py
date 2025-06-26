import pandas as pd
import time
import tracemalloc
import psutil
import os
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

RUNS = 1
metrics = {
    "accuracy": [],
    "elapsed_ms": [],
    "cpu_time_ms": [],
    "memory_kb": [],
    "allocated_memory_mb": []
}

# Читаем и подготавливаем данные
df = pd.read_csv("data/mushrooms_cleaned.csv")
X = df.drop("class", axis=1)
y = df["class"]

# Кодируем целевую переменную (если бинарная — метки 0 и 1)
le = LabelEncoder()
y_encoded = le.fit_transform(y)
num_classes = len(np.unique(y_encoded))

# Масштабируем признаки
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

for i in range(RUNS):
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

    # Преобразуем метки для Keras (one-hot для многоклассовой, но у тебя бинарная классификация, можно оставить целочисленные)
    if num_classes > 2:
        y_train_cat = to_categorical(y_train, num_classes)
        y_test_cat = to_categorical(y_test, num_classes)
    else:
        y_train_cat = y_train
        y_test_cat = y_test

    process = psutil.Process(os.getpid())
    cpu_start = process.cpu_times().user
    mem_start = process.memory_info().rss
    tracemalloc.start()
    start_time = time.time()

    # Создаём простую нейросеть
    model = Sequential()
    model.add(Dense(64, input_shape=(X_train.shape[1],), activation='relu'))
    model.add(Dense(32, activation='relu'))
    if num_classes > 2:
        model.add(Dense(num_classes, activation='softmax'))
        loss = 'categorical_crossentropy'
    else:
        model.add(Dense(1, activation='sigmoid'))
        loss = 'binary_crossentropy'

    model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])

    # Обучаем
    model.fit(X_train, y_train_cat, epochs=20, batch_size=32, verbose=0)

    # Предсказываем
    y_pred_prob = model.predict(X_test)
    if num_classes > 2:
        y_pred = np.argmax(y_pred_prob, axis=1)
    else:
        y_pred = (y_pred_prob > 0.5).astype(int).flatten()

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
print("Алгоритм: Keras Neural Network (среднее по", RUNS, "запускам)")
print(f"Accuracy: {sum(metrics['accuracy']) / RUNS:.4f}")
print(f"Время выполнения: {sum(metrics['elapsed_ms']) / RUNS:.2f} ms")
print(f"CPU время: {sum(metrics['cpu_time_ms']) / RUNS:.2f} ms")
print(f"Память: {sum(metrics['memory_kb']) / RUNS:.2f} Kb")
print(f"Аллоцировано памяти: {sum(metrics['allocated_memory_mb']) / RUNS:.2f} Mb")
