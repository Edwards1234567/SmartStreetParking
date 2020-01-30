# -*- coding: utf-8 -*-

import os
import subprocess
import streamlink
from enum import Enum


class VideoQuality(Enum):
    BEST = 'best'
    WORST = 'worst'
    Q1080 = '1080p'
    Q720 = '720p'
    Q320 = '320p'
    Q240 = '240p'
    Q144 = '144p'


def capture(url, output, quality=VideoQuality.BEST):
    livestream = [
        'streamlink',
        '-O',
        url,
        quality.value
    ]

    ffmpeg = [
        'ffmpeg',
        '-y',       # Force overwrite
        '-i',
        '-',
        '-f',
        'image2',
        '-vframes',
        '1',
        output
    ]

    p1 = subprocess.Popen(livestream, stderr=subprocess.DEVNULL, stdout=subprocess.PIPE)
    p2 = subprocess.Popen(ffmpeg, stdout=subprocess.DEVNULL,
                          stderr=subprocess.DEVNULL, stdin=p1.stdout)
    p1.stdout.close()
    p2.communicate()

    # Check output file exists
    if not os.path.isfile(output):
        raise IOError('Can not save image to this output path.')

    return output


def safe_capture(url, output, quality=VideoQuality.BEST):
    # Check video quality exists
    streams = streamlink.streams(url)
    if quality.value not in streams:
        msg = 'The specified stream(s) "{quality}" could not be found.'.format(
            quality=quality)
        raise ValueError(msg)

    # Check output path permission
    if not os.access(os.path.split(output)[0], os.W_OK):
        raise PermissionError('Can\'t write image to this path.')

    # Capture, capture, capture
    return capture(url, output, quality)
