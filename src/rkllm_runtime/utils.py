from typing import Any, ClassVar, Iterator, TypeVar, cast, dataclass_transform
from inspect import get_annotations

from ._rkllm import ffi


def fields(cls: type) -> Iterator[tuple[str, Any]]:
    """Yield the fields of a pseudo-dataclass.

    :param cls: the pseudo-dataclass
    :yield: a tuple of the field name and type
    """
    cls_annotations = get_annotations(cls)
    for name, _type in cls_annotations.items():
        if getattr(_type, "__origin__", None) is ClassVar:
            continue
        yield (name, _type)


# `typing.Self` type cannot be used in a metaclass
CStructType = TypeVar("CStructType", bound="CStructMeta")
T = TypeVar("T")


@dataclass_transform()
class CStructMeta(type):
    """
    Metaclass for constructing <class 'cffi.FFI.CData'> struct instances like Python dataclasses.

    The class must define annotations for all fields in the C struct.

    ## Example

    ```python
    class Pixel(metaclass=CStructMeta):
        r: int
        g: int
        b: int


    pixel = Pixel(r=255, g=0, b=0)
    print(pixel) # <cdata 'Pixel' owning 3 bytes>
    print(Pixel.format(pixel)) # Pixel{r=255, g=0, b=0}
    ```
    """

    _cname: str
    _cfields: dict[str, Any]
    _fields: dict[str, Any]

    def __new__(
        mcls: type[CStructType],
        name: str,
        bases: tuple[type, ...],
        namespace: dict[str, Any],
        /,
        *,
        cname: str | None = None,
        **kwargs: Any,
    ) -> CStructType:
        """Create a new class instance with C struct metadata.

        :param mcls: the metaclass
        :param name: the name of the new class
        :param bases: the base classes of the new class
        :param namespace: the namespace of the new class
        :param cname: the C struct name, defaults to the class name
        :raises TypeError: If the C type is not defined or not a struct
        :raises AttributeError: If there are missing or invalid annotations
        :return: the new class instance
        """
        if cname is None:
            cname = name

        try:
            ctype = ffi.typeof(cname)
        except ffi.error as e:
            raise TypeError(f"C type {cname!r} is not defined") from e
        if ctype.kind != "struct":
            raise TypeError(f"{ctype!r} is a {ctype.kind}, not a struct")

        cls = super().__new__(mcls, name, bases, namespace, **kwargs)
        cls._cname = ctype.cname
        cls._cfields = dict(ctype.fields)
        cls._fields = dict(fields(cls))

        if cls._cfields.keys() != cls._fields.keys():
            if missing_fields := cls._cfields.keys() - cls._fields.keys():
                raise AttributeError(
                    f"<class {cls.__name__}> is missing annotations for: {missing_fields}"
                )
            if invalid_fields := cls._fields.keys() - cls._cfields.keys():
                raise AttributeError(
                    f"<class {cls._cname}> defines invalid annotations: {invalid_fields}"
                )

        return cls

    def __call__(cls: type[T], *args: Any, **kwargs: Any) -> T:
        """Create a new <cdata 'struct'> instance.

        :param cls: a CStructMeta derived class
        :param args: positional arguments to initialize the struct
        :param kwargs: keyword arguments to initialize the struct
        :return: the new <cdata 'struct'> instance
        """
        # create a C struct instance
        obj = ffi.new(f"{cls._cname} *")  # type: ignore[attr-defined]
        # append positional arguments to keyword arguments
        kwargs |= {k: v for k, v in zip(cls._cfields.keys(), args)}  # type: ignore[attr-defined]

        # assign instance fields
        for attr, value in kwargs.items():
            # convert Python bytes to C char[]
            if cls._fields[attr] == bytes:  # type: ignore[attr-defined]
                setattr(obj, attr, ffi.new("char[]", value))
            # dereference C pointer
            elif (cls._fields[attr].__class__ is CStructMeta and  # type: ignore[attr-defined]
                  not cls._cfields[attr].type.cname.endswith("*")):  # type: ignore[attr-defined]
                setattr(obj, attr, value[0])
            # implicit conversion by CFFI
            else:
                setattr(obj, attr, value)

        return cast(T, obj)

    def format(cls: CStructType, obj: Any) -> str:
        """Format a <cdata 'struct'> instance as a string representation.

        :param cls: a CStructMeta derived class
        :param obj: a <cdata 'struct'> instance
        :return: a string representation of the object
        """
        return "%s{%s}" % (
            cls._cname,
            ", ".join(
                (f"{attr}={getattr(obj, attr)!r}" for attr in cls._cfields.keys())
            ),
        )


def from_dict(cls: type[T], data: dict[str, Any]) -> T:
    """Create a new instance of a pseudo-dataclass from a dictionary.

    :param cls: the pseudo-dataclass
    :param data: the dictionary of field values
    :return: the new instance of the pseudo-dataclass
    """
    cls_fields = dict(fields(cls))
    return cls(
        **{
            name: (
                _type
                if cls_fields[name].__module__ == "builtins"
                else from_dict(cls_fields[name], _type)
            )
            for name, _type in data.items()
        }
    )
