from setuptools import setup, find_packages

setup(
    name="lingua-gra-fractal-agi",
    version="0.1.0",
    description=(
        "Lingua GRA – фрактально-осознанный многоуровневый язык для "
        "самоулучшающихся AGI-агентов на основе GRA-обнулёнки и фрактальной геометрии эмбеддингов"
    ),
    author="AAA",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/Lingua-GRA-Fractal-AGI",
    license="MIT",
    packages=find_packages("src"),
    package_dir={"": "src"},
    include_package_data=True,
    install_requires=[
        "numpy>=1.22",
        "scipy>=1.8",
        "matplotlib>=3.5",
        "gensim>=4.3",
        "torch>=2.1",
        "tqdm>=4.65",
    ],
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
