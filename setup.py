"""Package setup for ReasonBrain."""

from pathlib import Path

from setuptools import find_packages, setup

ROOT = Path(__file__).parent

long_description = (ROOT / "README.md").read_text(encoding="utf-8")

setup(
    name="reasonbrain",
    version="0.1.0",
    description="Reproduction of ReasonBrain (ICML 2026) for hypothetical instruction-based image editing.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="ReasonBrain reproduction authors",
    license="Apache-2.0",
    python_requires=">=3.10",
    packages=find_packages(exclude=("tests", "scripts", "configs", "data")),
    include_package_data=True,
    install_requires=[
        # Heavy deps are pinned in requirements.txt to avoid resolver issues
        # when the package is installed in editable mode.
    ],
    entry_points={
        "console_scripts": [
            "reasonbrain-train=scripts.train:main",
            "reasonbrain-infer=scripts.infer:main",
            "reasonbrain-eval=scripts.evaluate:main",
        ],
    },
)
