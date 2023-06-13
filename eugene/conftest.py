import pytest

# In case pytest-nunit is not installed, defines a dummy fixture
try:
    import pytest_nunit
except ModuleNotFoundError:

    @pytest.fixture
    def add_nunit_attachment(request):
        def noop(file, description):
            pass

        return noop


# Command line options for pytest must be added from conftest.py from where
# `pytest` is called.
def pytest_addoption(parser):
    parser.addoption(
        "--internet-tests",
        action="store_true",
        default=False,
        help="Run tests that retrieve stuff from the internet. This increases test time.",
    )


def pytest_collection_modifyitems(config, items):
    run_internet = config.getoption("--internet-tests")
    skip_internet = pytest.mark.skip(reason="need --internet-tests option to run")
    for item in items:
        # All tests marked with `pytest.mark.internet` get skipped unless
        # `--run-internet` passed
        if not run_internet and ("internet" in item.keywords):
            item.add_marker(skip_internet)


# These fixtures provide a per test new copy of pbmc3k with some preprocessing run on it,
# without having to hit the disk or recompute normalization.
# The private fixture creates the object while the public one returns a deep copy.
@pytest.fixture(scope="session")
def _random1000():
    import eugene as eu

    return eu.datasets.random1000()


@pytest.fixture
def random1000(_random1000):
    return _random1000.copy()
