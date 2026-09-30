# Test Data

This directory contains test video data for the pixelforge examples.

## Files

- `test_frames.yuv` - Raw YUV420 frames (320x240, 30 frames)

## Regenerating Test Data

The test data was generated using FFmpeg:

```bash
# Generate raw YUV frames for encoding
ffmpeg -y -f lavfi -i "testsrc=duration=1:size=320x240:rate=30" \
    -pix_fmt yuv420p -f rawvideo test_frames.yuv
```

