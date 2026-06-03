import pytest

from src.datasets import (
    StandardScaler,
    make_binary_blobs,
    make_multiclass_blobs,
    make_two_moons,
    train_test_split,
)
from src.kernel_svm import BinaryKernelSVM
from src.pegasos import BinaryPegasosSVM, OneVsRestPegasosSVM


def test_binary_pegasos_reaches_reasonable_accuracy():
    X, y = make_binary_blobs(n_samples=1000, random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    model = BinaryPegasosSVM(lambda_=1e-3, epochs=20, batch_size=32, random_state=0)
    model.fit(X_train, y_train)
    assert model.score(X_test, y_test) > 0.85


def test_ovr_pegasos_multiclass_runs():
    X, y = make_multiclass_blobs(n_samples=900, n_classes=3, random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    model = OneVsRestPegasosSVM(lambda_=1e-3, epochs=20, batch_size=32, random_state=0)
    model.fit(X_train, y_train)
    assert model.score(X_test, y_test) > 0.75


def test_binary_pegasos_accepts_sparse_csr_input():
    sparse = pytest.importorskip("scipy.sparse")

    X, y = make_binary_blobs(n_samples=1000, random_state=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
    scaler = StandardScaler()
    X_train = sparse.csr_matrix(scaler.fit_transform(X_train))
    X_test = sparse.csr_matrix(scaler.transform(X_test))

    model = BinaryPegasosSVM(lambda_=1e-3, epochs=20, batch_size=32, random_state=1)
    model.fit(X_train, y_train)
    assert model.score(X_test, y_test) > 0.85


def test_rbf_kernel_svm_handles_nonlinear_data():
    X, y = make_two_moons(n_samples=300, noise=0.12, random_state=2)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = BinaryKernelSVM(C=8.0, gamma=1.3, max_passes=5, max_iter=80, random_state=2)
    model.fit(X_train, y_train)
    assert model.score(X_test, y_test) > 0.85
