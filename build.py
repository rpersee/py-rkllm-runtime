from pathlib import Path
import tomllib

from cffi import FFI


def skip_directives(*lines: str):
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


root_path = Path(__file__).parent
pyproject_file = root_path / "pyproject.toml"

with pyproject_file.open("rb") as file:
    pyproject = tomllib.load(file)
    package = pyproject["tool"]["poetry"]["packages"][0]

package_src = root_path / package["from"]
package_name = package["include"]

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
    ffibuilder.compile(str(package_src), verbose=True)
