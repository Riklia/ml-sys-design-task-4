import argparse

try:
    import cloudml_hypertune
except ImportError:
    cloudml_hypertune = None

import tensorflow as tf


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--hidden_units", type=int, default=128)
    parser.add_argument("--dropout_rate", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=4)
    return parser.parse_args()


def load_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    x_train = x_train[..., None]
    x_test = x_test[..., None]

    return (x_train[:12000], y_train[:12000]), (x_test[:3000], y_test[:3000])


def build_model(learning_rate, hidden_units, dropout_rate):
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(28, 28, 1)),
            tf.keras.layers.Conv2D(16, 3, activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(hidden_units, activation="relu"),
            tf.keras.layers.Dropout(dropout_rate),
            tf.keras.layers.Dense(10, activation="softmax"),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def report_metric(value, step):
    print(f"val_accuracy={value}")

    if cloudml_hypertune is None:
        return

    hypertune = cloudml_hypertune.HyperTune()
    hypertune.report_hyperparameter_tuning_metric(
        hyperparameter_metric_tag="val_accuracy",
        metric_value=value,
        global_step=step,
    )


def main():
    args = parse_args()

    (x_train, y_train), (x_val, y_val) = load_data()

    model = build_model(
        learning_rate=args.learning_rate,
        hidden_units=args.hidden_units,
        dropout_rate=args.dropout_rate,
    )

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=2,
    )

    val_accuracy = float(history.history["val_accuracy"][-1])
    report_metric(val_accuracy, args.epochs)


if __name__ == "__main__":
    main()
