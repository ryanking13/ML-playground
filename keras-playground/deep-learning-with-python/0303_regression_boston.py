import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
import numpy as np
from tensorflow import keras
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

MAX_DIM = 10000


def vectorize_seqences(sequences, dimension=MAX_DIM):
    results = np.zeros((len(sequences), dimension))
    for idx, s in enumerate(sequences):
        results[idx, s] = 1.0
    return results


def build_model(data):
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(64, input_shape=(data.shape[1],)))
    model.add(keras.layers.Activation("relu"))
    model.add(keras.layers.Dense(64))
    model.add(keras.layers.Activation("relu"))
    model.add(keras.layers.Dense(1))

    model.compile(
        optimizer=keras.optimizers.RMSprop(),
        loss=keras.losses.MeanSquaredError(),
        metrics=[keras.metrics.MeanAbsoluteError()],
    )
    return model


def smooth_curve(points, factor=0.9):
    smoothed = [points[0]]
    for p in points[1:]:
        prev = smoothed[-1]
        smoothed.append(prev * factor + p * (1 - factor))

    return smoothed


def main():
    (
        (train_data, train_labels),
        (test_data, test_labels),
    ) = keras.datasets.boston_housing.load_data()

    print(f"Raw Train Data Shape: {train_data.shape}")
    print(f"Raw Train Data Type: {train_data.dtype}")
    print(f"Raw Train Label Shape: {train_labels.shape}")
    print(f"Raw Train Label Type: {train_labels.dtype}")

    print(f"Raw Test Data Shape: {test_data.shape}")
    print(f"Raw Test Data Type: {test_data.dtype}")
    print(f"Raw Test Label Shape: {test_labels.shape}")
    print(f"Raw Test Label Type: {test_labels.dtype}")

    k = 4
    num_val = len(train_data) // k
    num_epochs = 100
    scores = []
    mae_histories = []
    for i in range(k):
        print(f"Fold {i}")
        val_data = train_data[i * num_val : (i + 1) * num_val]
        val_labels = train_labels[i * num_val : (i + 1) * num_val]

        train_data_partial = np.concatenate(
            [train_data[: i * num_val], train_data[(i + 1) * num_val :]], axis=0
        )
        train_labels_partial = np.concatenate(
            [train_labels[: i * num_val], train_labels[(i + 1) * num_val :]], axis=0,
        )

        # Data Standardization
        mean = np.mean(train_data_partial, axis=0)
        train_data_partial -= mean
        std = np.std(train_data_partial, axis=0)
        train_data_partial /= std

        val_data -= mean
        val_data /= std

        model = build_model(train_data_partial)
        history = model.fit(
            train_data_partial,
            train_labels_partial,
            epochs=num_epochs,
            batch_size=1,
            validation_data=(val_data, val_labels),
            verbose=0,
        )

        mae_history = history.history["val_mean_absolute_error"]
        mae_histories.append(mae_history)
        val_mse, val_mae = model.evaluate(val_data, val_labels, verbose=0)
        scores.append(val_mae)

    print(scores, np.mean(scores))
    avg_mae_history = [
        np.mean([x[i] for x in mae_histories]) for i in range(num_epochs)
    ]

    avg_mae_history_smoothed = smooth_curve(avg_mae_history[10:])

    df = pd.DataFrame(
        {
            "epochs": list(range(11, num_epochs + 1)),
            "ave_mae_history": avg_mae_history_smoothed,
        }
    )

    sns.lineplot(x="epochs", y="ave_mae_history", data=df)
    plt.show()

    # results = model.evaluate(x_test, y_test)
    # print(f"Loss: {results[0]}")
    # print(f"Accuracy: {results[1]}")

    # predictions = model.predict(x_test)
    # for p in predictions[:10]:
    #   print(np.argmax(p))


if __name__ == "__main__":
    main()
