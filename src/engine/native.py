import ctypes
import platform
from pathlib import Path

# one native library (engine + modules).
# every wrapper imports lib from here

_name = (
  "libchess.dylib" if platform.system() == "Darwin" else "libchess.so"
)
LIB_PATH = Path(__file__).resolve().parents[2] / "lib" / _name
if not LIB_PATH.exists():
  raise FileNotFoundError(f"{LIB_PATH} not found, run make")
lib = ctypes.CDLL(str(LIB_PATH))