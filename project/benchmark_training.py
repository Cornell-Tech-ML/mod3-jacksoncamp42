import time

import matplotlib.pyplot as plt
import numpy as np
from run_fast_tensor import FastTrain

import minitorch

FastTensorBackend = minitorch.TensorBackend(minitorch.FastOps)
GPUBackend = minitorch.TensorBackend(minitorch.CudaOps)


def timing_log_fn(start_time, times_per_epoch):
    def log_fn(epoch, total_loss, correct, losses):
        if epoch % 10 == 0 or epoch == max_epochs:
            time_per_epoch = (time.time() - start_time) / epoch if epoch > 0 else 0
            times_per_epoch.append(time_per_epoch)
            print(
                f"Epoch {epoch:3d} | Loss {total_loss:10.2f} | Correct {correct:4d} | Time/epoch {time_per_epoch:5.3f}s"
            )

    return log_fn


def run_training(
    dataset_name, backend_name, hidden_size=100, rate=0.05, pts=150, max_epochs=500
):
    print(f"\nTraining on {dataset_name} dataset with {backend_name.upper()} backend")
    print("=" * 80)

    # Select backend
    backend = GPUBackend if backend_name == "gpu" else FastTensorBackend

    # Create dataset
    if dataset_name == "simple":
        data = minitorch.datasets["Simple"].simple(pts)
    elif dataset_name == "xor":
        data = minitorch.datasets["Xor"](pts)
    else:  # split
        data = minitorch.datasets["Split"](pts)

    # Initialize model and timing
    start_time = time.time()
    times_per_epoch = []
    model = FastTrain(hidden_size, backend=backend)

    # Train
    model.train(
        data,
        learning_rate=rate,
        max_epochs=max_epochs,
        log_fn=timing_log_fn(start_time, times_per_epoch),
    )

    # Calculate final metrics
    total_time = time.time() - start_time
    avg_time_per_epoch = np.mean(times_per_epoch)

    # Calculate final accuracy
    X = minitorch.tensor(data.X, backend=backend)
    y = minitorch.tensor(data.y, backend=backend)
    out = model.model.forward(X).view(y.shape[0])
    y2 = minitorch.tensor(data.y)
    correct = int(((out.detach() > 0.5) == y2).sum()[0])
    accuracy = correct / len(data.y)

    return {
        "accuracy": accuracy,
        "avg_time_per_epoch": avg_time_per_epoch,
        "total_time": total_time,
        "times_per_epoch": times_per_epoch,
    }


def plot_results(results):
    datasets = ["Simple", "Xor", "Split"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot timing comparison
    cpu_times = [results[d]["cpu"]["avg_time_per_epoch"] for d in datasets]
    gpu_times = [results[d]["gpu"]["avg_time_per_epoch"] for d in datasets]

    x = np.arange(len(datasets))
    width = 0.35

    ax1.bar(x - width / 2, cpu_times, width, label="CPU")
    ax1.bar(x + width / 2, gpu_times, width, label="GPU")
    ax1.set_ylabel("Time per epoch (seconds)")
    ax1.set_title("Training Time Comparison")
    ax1.set_xticks(x)
    ax1.set_xticklabels(datasets)
    ax1.legend()

    # Plot accuracy comparison
    cpu_acc = [results[d]["cpu"]["accuracy"] for d in datasets]
    gpu_acc = [results[d]["gpu"]["accuracy"] for d in datasets]

    ax2.bar(x - width / 2, cpu_acc, width, label="CPU")
    ax2.bar(x + width / 2, gpu_acc, width, label="GPU")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Final Accuracy Comparison")
    ax2.set_xticks(x)
    ax2.set_xticklabels(datasets)
    ax2.legend()

    plt.tight_layout()
    plt.savefig("training_benchmark.png")
    plt.close()


if __name__ == "__main__":
    # Configuration
    HIDDEN = 100
    RATE = 0.05
    PTS = 150
    MAX_EPOCHS = 500

    results = {}
    datasets = ["simple", "xor", "split"]

    # Run benchmarks
    for dataset in datasets:
        results[dataset.capitalize()] = {
            "cpu": run_training(dataset, "cpu", HIDDEN, RATE, PTS, MAX_EPOCHS),
            "gpu": run_training(dataset, "gpu", HIDDEN, RATE, PTS, MAX_EPOCHS),
        }

    # Generate plot
    plot_results(results)

    # Print summary
    print("\nFinal Summary:")
    print("=" * 80)
    print(f"{'Dataset':<10} | {'Backend':<6} | {'Time/Epoch':<12} | {'Accuracy':<8}")
    print("-" * 80)

    for dataset in results:
        cpu_time = results[dataset]["cpu"]["avg_time_per_epoch"]
        gpu_time = results[dataset]["gpu"]["avg_time_per_epoch"]
        cpu_acc = results[dataset]["cpu"]["accuracy"]
        gpu_acc = results[dataset]["gpu"]["accuracy"]

        print(f"{dataset:<10} | {'CPU':<6} | {cpu_time:>8.3f}s   | {cpu_acc:>7.1%}")
        print(f"{' '*10} | {'GPU':<6} | {gpu_time:>8.3f}s   | {gpu_acc:>7.1%}")
        print("-" * 80)

    print("\nPlot saved as 'training_benchmark.png'")
