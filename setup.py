# Copyright © 2024 Apple Inc.

import sys
from pathlib import Path

from setuptools import setup

package_dir = Path(__file__).parent / "mlx_lm"
sys.path.append(str(package_dir))

from _version import __version__

MIN_MLX_VERSION = "0.29.2"

setup(
    name="mlx-streaming-llm",
    version=__version__,
    description="Streaming LLM generation with MLX - Implementation of StreamingLLM wait-k policy",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    readme="README.md",
    author_email="mlx@group.apple.com",
    author="MLX Contributors",
    url="https://github.com/ml-explore/mlx-lm",
    license="MIT",
    install_requires=[
        f"mlx>={MIN_MLX_VERSION}; platform_system == 'Darwin'",
        "numpy",
        "transformers>=4.39.3",
        "protobuf",
        "pyyaml",
        "jinja2",
    ],
    packages=["mlx_lm", "mlx_lm.models"],
    python_requires=">=3.8",
    extras_require={
        "test": ["datasets"],
        "cuda": [f"mlx[cuda]>={MIN_MLX_VERSION}"],
        "cpu": [f"mlx[cpu]>={MIN_MLX_VERSION}"],
    },
    entry_points={
        "console_scripts": []
    },
)
