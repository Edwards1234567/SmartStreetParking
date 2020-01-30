# -*- coding: utf-8 -*-

import shutil
import importlib.util

# Check requirement python package and ffmpeg
if not importlib.util.find_spec('streamlink'):
    raise ImportError('Please install streamlink first.')
if not shutil.which('ffmpeg'):
    raise OSError('Please install ffmpeg first.')

# Expose API
from pylivecap.core import capture, safe_capture, VideoQuality
