from pathlib import Path
import platform

from setuptools import Extension, find_packages, setup

PROJECT_PATH = Path(__file__).resolve().parent
SOURCE_PATH = PROJECT_PATH / "source"
DATA_PATH = PROJECT_PATH / "data"

file_required = PROJECT_PATH / "requirements.txt"
with file_required.open() as file_:
    install_requires = file_.read().splitlines()

# Include data files in the package
package_data = {}
for data_file in DATA_PATH.rglob("*"):
    if data_file.is_file():
        relative_path = data_file.relative_to(DATA_PATH)
        package_data["MulensModel"] = [f"data/{relative_path}"]

version = "unknown"
with Path(SOURCE_PATH / "MulensModel" / "version.py").open() as in_put:
    for line_ in in_put.readlines():
        if line_.startswith('__version__'):
            version = line_.split()[2][1:-1]

source_VBBL = SOURCE_PATH / "VBBL"
source_AC = SOURCE_PATH / "AdaptiveContouring"
source_MM = SOURCE_PATH / "MulensModel"
source_MMmo = source_MM / "mulensobjects"

# C/C++ Extensions
kwargs = dict()
if platform.system().upper() != "WINDOWS":
    kwargs['libraries'] = ["m"]
ext_AC = Extension(
    "MulensModel.AdaptiveContouring", **kwargs,
    sources=[str(f.relative_to(PROJECT_PATH)) for f in source_AC.glob("*.c")])
ext_VBBL = Extension(
    "MulensModel.VBBL", **kwargs,
    sources=[
        str(f.relative_to(PROJECT_PATH)) for f in source_VBBL.glob("*.cpp")])

setup(
    name='MulensModel',
    version=version,
    url='https://github.com/rpoleski/MulensModel',
    project_urls={
        'documentation': 'https://github.com/rpoleski/MulensModel'},
    ext_modules=[ext_AC, ext_VBBL],
    author='Radek Poleski & Jennifer Yee',
    author_email='radek.poleski@gmail.com',
    description='package for modeling gravitational microlensing events',
    long_description='package for modeling gravitational microlensing events',
    packages=find_packages(where="source"),
    package_dir={"": "source"},
    include_package_data=True,
    package_data=package_data,
    python_requires=">=3.6",
    install_requires=install_requires,
)
