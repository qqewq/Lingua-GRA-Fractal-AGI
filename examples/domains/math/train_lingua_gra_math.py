from lingua_gra.domains.math_domain import default_math_domain, default_math_config
# и твои существующие модули: fractal_utils, training и т.п.

def main():
    data_path = "examples/domains/math/data"
    domain = default_math_domain(data_path)
    config = default_math_config(d2_target=None)
    print("Domain:", domain)
    print("Config:", config)
    # TODO: сюда потом подключишь реальное обучение через training.py

if __name__ == "__main__":
    main()
