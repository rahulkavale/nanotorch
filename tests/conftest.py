import pytest

from nanotorch.scenarios import get_scenario


# Scenarios are fixtures so tests read as "what behavior" rather than
# "how to build data". This keeps the test intent focused and avoids
# repeating setup across files.


@pytest.fixture
def scenario_single_point():
    return get_scenario("single_point")


@pytest.fixture
def scenario_multi_point_no_bias():
    return get_scenario("multi_point_no_bias")


@pytest.fixture
def scenario_with_bias():
    return get_scenario("with_bias")


@pytest.fixture
def scenario_constant_target():
    return get_scenario("constant_target")


@pytest.fixture
def scenario_noisy_linear():
    return get_scenario("noisy_linear")
