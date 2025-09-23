import numpy as np
import pytest
from skimage.io import imsave

from src.train_models import ChannelMixer, CellPaintingDataset, ImageSample


def write_tif(path, seed):
    rng = np.random.default_rng(seed)
    array = (rng.random((16, 16)) * 65535).astype(np.uint16)
    imsave(path, array)


def test_channel_mixer_softmax():
    mixer = ChannelMixer(in_channels=3)
    output = mixer.importances()
    assert np.isclose(output.sum(), 1.0)
    assert (output > 0).all()


def test_cellpainting_dataset_returns_tensor(tmp_path):
    paths = []
    for idx in range(3):
        file_path = tmp_path / f"image_{idx:03d}.tif"
        write_tif(file_path, seed=idx)
        paths.append(file_path)

    sample = ImageSample(paths=paths, plate="Week1", compound_id="000001", well="A01", site="00", label=1.0)
    dataset = CellPaintingDataset([sample], image_size=8)
    tensor, label, meta = dataset[0]
    assert tensor.shape == (3, 8, 8)
    assert 0.0 <= tensor.max() <= 1.0
    assert pytest.approx(label.item(), 1.0)
    assert meta.compound_id == "000001"
