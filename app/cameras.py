"""Which cameras does this computer have, and what are they called?

OpenCV opens a camera by its number (0, 1, 2...) but can't tell its name. Windows can: camera_names()
asks DirectShow, the part of Windows that OpenCV opens cameras with first (CAMERA_BACKENDS in
app/webcam.py). DirectShow lists the cameras in the same order OpenCV numbers them, so the first name
is camera 0, the second is camera 1, and so on. Nothing is turned on to find them.

The code below talks to Windows' COM system directly through ctypes. It is low-level, and you don't
need to follow it to understand the rest of the project: all it does is fill a list of names.
"""

import ctypes
import sys
import threading
from ctypes import POINTER, byref, c_ulong, c_ushort, c_void_p, c_wchar_p

# The COM objects and interfaces used, by their Windows IDs.
SYSTEM_DEVICE_ENUM = "{62BE5D10-60EB-11d0-BD3B-00A0C911CE86}"  # lists devices by category
CREATE_DEV_ENUM = "{29840822-5B84-11D0-BD3B-00A0C911CE86}"
VIDEO_INPUT_DEVICE_CATEGORY = "{860BB310-5D01-11d0-BD3B-00A0C911CE86}"  # cameras, as DirectShow sees them
PROPERTY_BAG = "{55272A00-42CB-11CE-8135-00AA004BB851}"  # where a device keeps its name
VT_BSTR = 8  # "this value is text"


def camera_names() -> list[str]:
    """The names of the cameras, in OpenCV's order. Empty if Windows can't be asked (other systems)."""
    if sys.platform != "win32":
        return []
    names = []

    def ask_windows():
        try:
            names.extend(directshow_camera_names())
        except OSError as error:
            print(f"Could not list the cameras: {error}")

    # COM is set up and taken down in a thread of its own, so it can't disturb the window's thread.
    worker = threading.Thread(target=ask_windows, daemon=True)
    worker.start()
    worker.join(5)
    return names if not worker.is_alive() else []


class GUID(ctypes.Structure):
    _fields_ = [("data1", ctypes.c_uint32), ("data2", ctypes.c_uint16), ("data3", ctypes.c_uint16),
                ("data4", ctypes.c_ubyte * 8)]


class VARIANT(ctypes.Structure):
    _fields_ = [("vt", c_ushort), ("reserved", c_ushort * 3), ("value", c_void_p), ("value2", c_void_p)]


def guid(text: str) -> GUID:
    result = GUID()
    ctypes.oledll.ole32.CLSIDFromString(c_wchar_p(text), byref(result))
    return result


def call(interface: c_void_p, index: int, argtypes: tuple, *args, restype=ctypes.HRESULT):
    """Call method number `index` of a COM interface (methods are numbered in the order Windows declares them)."""
    methods = ctypes.cast(interface, POINTER(POINTER(c_void_p))).contents
    return ctypes.WINFUNCTYPE(restype, c_void_p, *argtypes)(methods[index])(interface, *args)


def release(interface: c_void_p) -> None:
    if interface:
        call(interface, 2, (), restype=c_ulong)  # IUnknown::Release


def directshow_camera_names() -> list[str]:
    ctypes.oledll.ole32.CoInitializeEx(None, 0)
    names = []
    device_enum, cameras = c_void_p(), c_void_p()
    try:
        ctypes.oledll.ole32.CoCreateInstance(
            byref(guid(SYSTEM_DEVICE_ENUM)), None, 1, byref(guid(CREATE_DEV_ENUM)), byref(device_enum)
        )
        # ICreateDevEnum::CreateClassEnumerator: 0 means "found some", 1 means "none at all".
        found = call(device_enum, 3, (POINTER(GUID), POINTER(c_void_p), c_ulong),
                     byref(guid(VIDEO_INPUT_DEVICE_CATEGORY)), byref(cameras), 0)
        while found == 0 and cameras:
            camera, fetched = c_void_p(), c_ulong()
            # IEnumMoniker::Next gives the next camera; anything but 0 means there are no more.
            if call(cameras, 3, (c_ulong, POINTER(c_void_p), POINTER(c_ulong)), 1, byref(camera), byref(fetched)) != 0:
                break
            names.append(read_name(camera) or f"Camera {len(names)}")  # a camera without a name still counts
            release(camera)
    finally:
        release(cameras)
        release(device_enum)
        ctypes.windll.ole32.CoUninitialize()
    return names


def read_name(camera: c_void_p) -> str | None:
    bag, value = c_void_p(), VARIANT()
    try:
        # IMoniker::BindToStorage opens the camera's property bag; IPropertyBag::Read reads its name.
        call(camera, 9, (c_void_p, c_void_p, POINTER(GUID), POINTER(c_void_p)),
             None, None, byref(guid(PROPERTY_BAG)), byref(bag))
        call(bag, 3, (c_wchar_p, POINTER(VARIANT), c_void_p), "FriendlyName", byref(value), None)
        name = ctypes.wstring_at(value.value) if value.vt == VT_BSTR and value.value else None
        ctypes.oledll.oleaut32.VariantClear(byref(value))
        return name
    except OSError:
        return None
    finally:
        release(bag)
