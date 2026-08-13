"""pytest 根配置：确保 safe_motion 包可被导入。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
