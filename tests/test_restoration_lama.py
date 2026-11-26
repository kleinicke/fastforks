"""Regression checks for Big LaMa's Fourier transforms and CUDA graph results."""

from __future__ import annotations

import pytest
import torch

from fastforks import restoration
from fastforks.kernels import cudnn_hacc
from fastforks.restoration.models.lama.model import Kernels
from fastforks.restoration.models.lama.net._export import inverse_real_dft2, real_dft2


@pytest.mark.parametrize(("height", "width"), [(3, 5), (8, 10), (32, 48), (64, 64)])
def test_fourier_export_matches_fft(height: int, width: int) -> None:
    """Both transforms agree with torch.fft, including learned spectra with arbitrary bins."""
    generator = torch.Generator().manual_seed(43)
    signal = torch.randn(2, 3, height, width, generator=generator)
    actual = real_dft2(signal)
    expected = torch.view_as_real(torch.fft.rfft2(signal, norm="ortho"))
    torch.testing.assert_close(actual, expected, atol=6e-5, rtol=1e-5)

    spectrum = torch.randn(2, 3, height, width // 2 + 1, 2, generator=generator)
    actual = inverse_real_dft2(spectrum, width)
    expected = torch.fft.irfft2(torch.view_as_complex(spectrum), s=(height, width), norm="ortho")
    torch.testing.assert_close(actual, expected, atol=6e-5, rtol=1e-5)


@pytest.mark.slow
def test_fourier_export_runs_with_onnxruntime(tmp_path) -> None:
    """Runtime support includes the fixed Fourier bases and a nontrivial transformed spectrum."""
    ort = pytest.importorskip("onnxruntime")
    pytest.importorskip("onnxscript")

    class Transform(torch.nn.Module):
        def forward(self, image, gain):
            return inverse_real_dft2(real_dft2(image) * gain, image.shape[-1])

    generator = torch.Generator().manual_seed(1)
    gain = torch.randn(1, 1, 5, 4, 2, generator=generator)
    trace = torch.randn(1, 1, 5, 7, generator=generator)
    path = tmp_path / "fourier.onnx"
    torch.onnx.export(Transform().eval(), (trace, gain), path, dynamo=True, input_names=["image", "gain"])
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])

    image = torch.randn(1, 1, 5, 7, generator=generator)
    spectrum = torch.view_as_real(torch.fft.rfft2(image, norm="ortho")) * gain
    expected = torch.fft.irfft2(torch.view_as_complex(spectrum), s=(5, 7), norm="ortho")
    (actual,) = session.run(None, {"image": image.numpy(), "gain": gain.numpy()})
    torch.testing.assert_close(torch.from_numpy(actual), expected, atol=1e-5, rtol=1e-5)


@pytest.mark.gpu
@pytest.mark.model
@pytest.mark.usefixtures("graph_lifecycle")
def test_graph_outputs_are_cropped_and_owned() -> None:
    """A single padded graph can serve different crops without changing earlier results."""
    model = restoration.load("big-lama", device="cuda", mode="kernels")
    assert isinstance(model._runner, Kernels)
    graphs = model._runner.graphs
    assert graphs is not None
    first, first_mask = model.example_inputs(65, 79)
    second, second_mask = model.example_inputs(67, 77)
    expected = model(first, mask=first_mask).image
    for _ in range(cudnn_hacc.capture_after() - 1):
        model(first, mask=first_mask)
    retained = model(first, mask=first_mask).image
    assert graphs.graphs
    changed = model(1 - second, mask=second_mask).image
    assert len(graphs.graphs) == 1
    model.release_graphs()
    assert torch.equal(changed, model(1 - second, mask=second_mask).image)
    assert torch.equal(retained, expected)
