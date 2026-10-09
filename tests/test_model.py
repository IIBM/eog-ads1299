import numpy as np

from eog_app.model import predict, train_model


def test_model_training_saves_predictable_weights(tmp_path) -> None:
    features = np.array([[-2.0, -1.0], [-1.0, -2.0], [1.0, 2.0], [2.0, 1.0]])
    positions = np.array([[0.1, 0.2], [0.2, 0.1], [0.8, 0.9], [0.9, 0.8]])
    model_path = tmp_path / "model.npz"

    loss = train_model(features, positions, model_path, epochs=25)
    output = predict(features, model_path)

    assert model_path.exists()
    assert np.isfinite(loss)
    assert output.shape == positions.shape
    assert np.all((output >= 0.0) & (output <= 1.0))
