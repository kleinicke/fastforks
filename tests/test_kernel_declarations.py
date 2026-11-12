"""Dependency checks distinguish optional shared implementations from actual launches."""

import pytest

from test_golden import _declared_launches
from test_policy import _kernel_import_bounds


def test_unused_shared_backend_is_not_a_required_dependency():
    _kernel_import_bounds("cnn", {"fvm_conv"}, {"fvm_conv"}, {"fvm_conv", "vit_metal"})


def test_direct_kernel_import_must_be_declared():
    with pytest.raises(AssertionError, match=r"direct kernel imports missing.*fvm_conv"):
        _kernel_import_bounds("cnn", set(), {"fvm_conv"}, {"fvm_conv"})


def test_declared_library_must_have_an_import_path():
    with pytest.raises(AssertionError, match=r"absent from import closure.*unknown"):
        _kernel_import_bounds("cnn", {"unknown"}, set(), {"fvm_conv"})


@pytest.mark.parametrize("library", ["fvm_conv", "cudnn_hacc"])
def test_undeclared_launched_library_fails(library):
    with pytest.raises(AssertionError, match=rf"launched undeclared kernel libraries.*{library}"):
        _declared_launches("example@1", set(), {library})


def test_declared_but_unexercised_geometry_is_allowed():
    _declared_launches("example@1", {"fvm_conv", "cudnn_hacc"}, {"fvm_conv"})
