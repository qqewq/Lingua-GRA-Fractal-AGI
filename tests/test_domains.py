from lingua_gra.domains.math_domain import default_math_domain, default_math_config

def test_math_domain_and_config():
    domain = default_math_domain("examples/domains/math/data")
    config = default_math_config(d2_target=7.5)
    assert domain.name == "math"
    assert "semantic" in config.levels
    assert config.d2_target == 7.5
