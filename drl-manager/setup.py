from setuptools import setup, find_packages

setup(
    name="gym_cloudsimplus_lb",
    version="1.0.0",  # Updated for reorganized structure
    description="Gymnasium environments for CloudSimPlus-based cloud scheduling with RL",
    author="DRL Cloud Scheduling Team",
    packages=find_packages(exclude=["tests", "scripts", "mnt"]),  # Include gym_cloudsimplus and src packages
    install_requires=[
        "gymnasium>=0.29.0",
        "py4j>=0.10.9",
        "numpy>=1.24.0",
        "torch>=2.0.0",
        "stable-baselines3>=2.0.0",
        "sb3-contrib>=2.0.0",
        "pyyaml>=6.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
        ]
    },
    python_requires='>=3.10',  # Adjusted for broader compatibility
    entry_points={
        'console_scripts': [
            'drl-train=entrypoint:main',
        ],
    },
)
