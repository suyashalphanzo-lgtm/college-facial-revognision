import numpy as np

from app.face_engine import TEMPLATE_POINTS, align_face, similarity_transform


def face_row(points):
    """A detected face as the detector gives it: box, 5 points, confidence."""
    return np.concatenate([[0, 0, 10, 10], np.asarray(points).ravel(), [0.9]]).astype(np.float32)


def test_a_face_already_on_the_template_comes_back_unchanged():
    image = np.random.default_rng(0).integers(0, 256, (112, 112, 3), dtype=np.uint8)
    assert np.array_equal(align_face(image, face_row(TEMPLATE_POINTS)), image)


def test_a_turned_bigger_moved_face_is_put_back_on_the_template():
    angle = np.radians(30)
    turn = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    points = 2.5 * TEMPLATE_POINTS @ turn.T + [300, 150]

    matrix = similarity_transform(points, TEMPLATE_POINTS)
    moved = points @ matrix[:, :2].T + matrix[:, 2]

    assert np.allclose(moved, TEMPLATE_POINTS, atol=1e-6)


def test_alignment_only_turns_never_mirrors():
    mirrored_points = TEMPLATE_POINTS * [-1, 1] + [112, 0]  # right and left swapped, like a mirror image
    matrix = similarity_transform(mirrored_points, TEMPLATE_POINTS)
    assert np.linalg.det(matrix[:, :2]) > 0
