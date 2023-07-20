from setuptools import setup

if __name__ == "__main__":
    install_requires = [] 
    with open("requirements.txt") as f:
        install_requires = f.read().splitlines()
    setup(install_requires=install_requires)