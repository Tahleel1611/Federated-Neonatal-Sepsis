import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

_here = os.path.dirname(os.path.abspath(__file__))
_app = os.path.join(_here, "streamlit_app.py")

with open(_app, "r", encoding="utf-8") as _f:
    exec(compile(_f.read(), _app, "exec"), {"__file__": _app, "__name__": "__main__"})
