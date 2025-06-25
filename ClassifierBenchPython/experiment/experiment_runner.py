from statistics import mean
from experiment.performance_measurer import PerformanceMeasurer

class ExperimentRunner:
    def __init__(self, x_train, x_test, y_train, y_test):
        self.X_train = x_train
        self.X_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.measurer = PerformanceMeasurer()

    def run_experiment(self, algorithm_runner, runs_count=5):
        elapsed_times = []
        cpu_times = []
        memory_kbs = []
        allocated_mbs = []
        all_metrics = []

        for _ in range(runs_count):
            def wrapped():
                algorithm_runner.train(self.X_train, self.y_train)
                return algorithm_runner.predict(self.X_test)

            predictions, perf = self.measurer.measure(wrapped)
            metrics = algorithm_runner.evaluate(self.y_test, predictions)

            elapsed_times.append(perf["elapsed_ms"])
            cpu_times.append(perf["cpu_time_ms"])
            memory_kbs.append(perf["memory_kb"])
            allocated_mbs.append(perf["allocated_memory_mb"])
            all_metrics.append(metrics)

        avg_metrics = self._average_dicts(all_metrics)

        return {
            "elapsed_ms": mean(elapsed_times),
            "cpu_time_ms": mean(cpu_times),
            "memory_kb": mean(memory_kbs),
            "allocated_memory_mb": mean(allocated_mbs),
            "metrics": avg_metrics
        }

    def _average_dicts(self, dicts):
        from collections import defaultdict
        agg = defaultdict(list)
        for d in dicts:
            for k, v in d.items():
                agg[k].append(v)
        return {k: mean(vs) for k, vs in agg.items()}
