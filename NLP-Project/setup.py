from setuptools import setup, find_packages

setup(
    name="whisper-transcriber",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        line.strip() for line in open("requirements.txt") if line.strip()
    ],
    entry_points={
        'console_scripts': [
            'whisper-transcribe=transcriber.cli:main',
        ],
    },
    author="Your Name",
    description="Multilingual audio/video transcription using Whisper",
    python_requires=">=3.8",
)