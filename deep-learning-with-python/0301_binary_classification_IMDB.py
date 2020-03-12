import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
import numpy as np
from tensorflow import keras
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


MAX_DIM = 10000


def indices2sentence(indices):
    word_index_dict = keras.datasets.imdb.get_word_index()
    index_word_dict = {v: k for k, v in word_index_dict.items()}

    return " ".join(index_word_dict.get(idx - 3, "") for idx in indices)


def vectorize_seqences(sequences, dimension=MAX_DIM):
    results = np.zeros((len(sequences), dimension))
    for idx, s in enumerate(sequences):
        results[idx, s] = 1.0
    return results


def main():
    (
        (train_data, train_labels),
        (test_data, test_labels),
    ) = keras.datasets.imdb.load_data(num_words=MAX_DIM)

    # print(indices2sentence(train_data[0]))
    print(f"Raw Train Data Shape: {train_data.shape}")
    print(f"Raw Train Data Type: {train_data.dtype}")
    print(f"Raw Train Label Shape: {train_labels.shape}")
    print(f"Raw Train Label Type: {train_labels.dtype}")

    print(f"Raw Test Data Shape: {test_data.shape}")
    print(f"Raw Test Data Type: {test_data.dtype}")
    print(f"Raw Test Label Shape: {test_labels.shape}")
    print(f"Raw Test Label Type: {test_labels.dtype}")

    x_train = vectorize_seqences(train_data)
    x_test = vectorize_seqences(test_data)
    y_train = np.asarray(train_labels).astype("float32")
    y_test = np.asarray(test_labels).astype("float32")

    print(f"Vectorized Train Data Shape: {x_train.shape}")
    print(f"Vectorized Train Data Type: {x_train.dtype}")
    print(f"Vectorized Train Label Shape: {y_train.shape}")
    print(f"Vectorized Train Label Type: {y_train.dtype}")

    print(f"Vectorized Test Data Shape: {x_test.shape}")
    print(f"Vectorized Test Data Type: {x_test.dtype}")
    print(f"Vectorized Train Label Shape: {y_test.shape}")
    print(f"Vectorized Train Label Type: {y_test.dtype}")

    val_size = 10000
    x_val = x_train[:val_size]
    x_train_partial = x_train[val_size:]
    y_val = y_train[:val_size]
    y_train_partial = y_train[val_size:]

    model = keras.models.Sequential()
    model.add(keras.layers.Dense(16, input_shape=(10000,)))
    model.add(keras.layers.Activation("relu"))
    model.add(keras.layers.Dense(16))
    model.add(keras.layers.Activation("relu"))
    model.add(keras.layers.Dense(1))
    model.add(keras.layers.Activation("sigmoid"))

    # print(model.summary())

    model.compile(
        optimizer=keras.optimizers.RMSprop(),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=[keras.metrics.BinaryAccuracy()],
    )

    history = model.fit(
        x_train_partial,
        y_train_partial,
        epochs=4,
        batch_size=512,
        validation_data=(x_val, y_val),
    )

    history_df = pd.DataFrame(history.history)
    history_df["epochs"] = list(range(1, len(history.history["loss"]) + 1))
    sns.lineplot(
        x="epochs",
        y="value",
        hue="variable",
        data=pd.melt(history_df, id_vars=["epochs"]),
    )

    plt.show()

    results = model.evaluate(x_test, y_test)
    print(f"Loss: {results[0]}")
    print(f"Accuracy: {results[1]}")

    print(model.predict(x_test))


if __name__ == "__main__":
    main()
