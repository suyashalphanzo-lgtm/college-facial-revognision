"""Checks for our own SFace (training/): the network's shape, the SFace loss's behaviour and the
webcam-style photo changes. They need PyTorch, but no GPU and no datasets."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from training.datasets import webcam_style  # noqa: E402
from training.mobilefacenet import MobileFaceNet  # noqa: E402
from training.sface_loss import SFaceLoss  # noqa: E402


def test_network_turns_each_face_into_128_numbers():
    faces = torch.rand(2, 3, 112, 112) * 255
    assert MobileFaceNet().eval()(faces).shape == (2, 128)


def loss_and_gradient(embedding, label):
    """SFace loss with 3 people whose centres point along the 3 axes."""
    loss_function = SFaceLoss(embedding_size=3, num_people=3)
    with torch.no_grad():
        loss_function.centres.copy_(torch.eye(3))
    embedding = torch.tensor([embedding], requires_grad=True)
    loss, _ = loss_function(embedding, torch.tensor([label]))
    loss.backward()
    return loss.item(), embedding.grad.abs().max().item()


def test_a_face_already_close_to_its_own_person_is_left_alone():
    # Angle 0 to its own centre (below a) and 90 degrees to the others (above b): both sigmoid
    # weights are about 0, so there is nothing left to learn from this face.
    loss, gradient = loss_and_gradient([1.0, 0.0, 0.0], label=0)
    assert abs(loss) < 1e-6
    assert gradient < 1e-6


def test_a_face_that_looks_like_someone_else_is_pushed_away():
    # Labelled person 0 but pointing almost at person 1's centre: pushed from 1 and pulled towards 0.
    loss, gradient = loss_and_gradient([0.2, 1.0, 0.0], label=0)
    assert loss > 10
    assert gradient > 1


def test_webcam_style_changes_keep_the_size_and_leave_many_photos_alone():
    np.random.seed(0)
    face = (np.random.rand(112, 112, 3) * 255).astype(np.uint8)
    results = [webcam_style(face) for _ in range(200)]
    assert all(result.shape == (112, 112, 3) and result.dtype == np.uint8 for result in results)
    # No change happens with chance 0.8 * 0.8 * 0.7 = 45%.
    unchanged = sum(np.array_equal(result, face) for result in results)
    assert 60 < unchanged < 120
