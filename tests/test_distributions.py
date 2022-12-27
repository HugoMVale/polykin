from polykin.distributions import Flory, Poisson


def test_init():
    dist = {"flory": Flory(), "poisson": Poisson()}
    for item in dist.values():
        assert item(100) > 0
