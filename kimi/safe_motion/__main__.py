"""支持 python -m safe_motion scenarios/xxx.json 调用方式。"""
import sys

from .replay import main

if __name__ == "__main__":
    sys.exit(main())
