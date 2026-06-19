from setuptools import find_packages, setup

with open("requirements.txt") as f:
    install_requires = [
        line.strip()
        for line in f
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="contextgc",
    version="0.1.0",
    description="Context window garbage collector for LLM agentic pipelines",
    packages=find_packages(),
    install_requires=install_requires,
    python_requires=">=3.9",
)
