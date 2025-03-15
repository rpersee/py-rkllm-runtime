from collections.abc import Iterator
from pathlib import Path
import configparser
import os
import shutil
import tomllib

from cffi import FFI


def skip_directives(*lines: str) -> Iterator[str]:
    skip = False
    for line in lines:
        if line.startswith("#"):
            match line.split():
                case ["#ifdef", symbole]:
                    skip = True
                case ["#endif"]:
                    skip = False
            continue
        if skip:
            continue
        yield line


def preprocess_csource(text: str) -> str:
    return "\n".join(skip_directives(*text.splitlines()))


def pyproject_packages(base_dir: Path) -> Iterator[tuple[Path, str]]:
    pyproject_file = base_dir / "pyproject.toml"
    with pyproject_file.open("rb") as file:
        pyproject = tomllib.load(file)

    for package in pyproject["tool"]["poetry"]["packages"]:
        yield (base_dir / package["from"], package["include"])


def env_config(env_prefix: str = "RKLLMRT_BUILD") -> configparser.SectionProxy:
    env_config = configparser.ConfigParser()
    env_config.read_dict(
        {
            env_prefix: {
                key.removeprefix(f"{env_prefix}_").lower(): value
                for key, value in os.environ.items()
                if key.startswith(env_prefix)
            }
        }
    )
    return env_config[env_prefix]


root_path = Path(__file__).parents[1]
package_from, package_name = next(pyproject_packages(root_path))

library_path = root_path / "lib"
include_path = root_path / "include"

header_file = include_path / "rkllm.h"
header_source = preprocess_csource(header_file.read_text())

ffibuilder = FFI()
ffibuilder.cdef(header_source)
ffibuilder.cdef(
    """
    extern "Python" void rkllm_result_callback(RKLLMResult* result, void* userdata, LLMCallState state);
    """
)

ffibuilder.set_source(
    f"{package_name}._rkllm",
    """
    #include <stdbool.h>
    #include "rkllm.h"
    """,
    libraries=["rkllmrt"],
    library_dirs=[str(library_path)],
    include_dirs=[str(include_path)],
)

if __name__ == "__main__":
    config = env_config()

    ffibuilder.compile(str(package_from), verbose=config.getboolean("verbose", True))

    if config.getboolean("include_lib", False):
        package_dir = package_from / package_name
        for file in library_path.glob("*.so"):
            shutil.copy(file, package_dir)
