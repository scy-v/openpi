import numpy as np

from openpi.models import model as _model
from openpi.policies import dual_ur_policy


def test_pi05_uses_right_wrist_image():
    right_wrist_image = np.full((224, 224, 3), 127, dtype=np.uint8)
    data = {
        "observation/state": np.zeros(38, dtype=np.float32),
        "observation/exterior_image": np.zeros((224, 224, 3), dtype=np.uint8),
        "observation/left_wrist_image": np.zeros((224, 224, 3), dtype=np.uint8),
        "observation/right_wrist_image": right_wrist_image,
    }

    inputs = dual_ur_policy.DualURInputs(model_type=_model.ModelType.PI05)(data)

    assert inputs["image_mask"]["right_wrist_0_rgb"]
    np.testing.assert_array_equal(inputs["image"]["right_wrist_0_rgb"], right_wrist_image)
