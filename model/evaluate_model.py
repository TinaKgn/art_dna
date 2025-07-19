from model.data_loader import get_data_generators
from tensorflow.keras.models import load_model


def evaluate_model():
    train_gen, val_gen, test_gen = get_data_generators(
        "raw_data/labeled_data.csv", include_test=True
    )

    model = load_model("model/art_style_classifier.keras")

    print(f"🧪 Evaluating on test set...")
    test_loss, test_accuracy = model.evaluate(test_gen)

    print(f"✅ Test accuracy: {test_accuracy:.4f}")
    print(f"📊 Test loss: {test_loss:.4f}")


if __name__ == "__main__":
    evaluate_model()
