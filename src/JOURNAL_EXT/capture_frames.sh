#!/bin/bash

# Check if ffmpeg is installed
if ! command -v ffmpeg &> /dev/null; then
    echo "ffmpeg is not installed. Please install ffmpeg."
    exit 1
fi

# Default settings
DEVICE="/dev/video0"    # Video device
FORMAT="mjpeg"          # Default to MJPG
FRAMES=30               # Number of frames
FRAMERATE=30            # Frame rate (fps)
RESOLUTION="1920x1080"  # Resolution (WIDTHxHEIGHT)

# Parse command-line options
while [ "$1" != "" ]; do
    case $1 in
        --help )    echo "Usage: $0 [options]"
                    echo "  --device DEV     Video device (default: /dev/video0)"
                    echo "  --format FMT     Format: mjpeg or yuyv422 (default: mjpeg)"
                    echo "  --frames N       Number of frames (default: 90)"
                    echo "  --framerate FPS  Frame rate (default: 30)"
                    echo "  --resolution WxH Resolution (default: 1920x1080)"
                    exit 0
                    ;;
        --device )  shift; DEVICE=$1 ;;
        --format )  shift; FORMAT=$1 ;;
        --frames )  shift; FRAMES=$1 ;;
        --framerate ) shift; FRAMERATE=$1 ;;
        --resolution ) shift; RESOLUTION=$1 ;;
        * )         echo "Unknown option: $1"; exit 1 ;;
    esac
    shift
done

# Validate inputs
if [ ! -e "$DEVICE" ]; then
    echo "Device $DEVICE does not exist"
    exit 1
fi
if [ "$FORMAT" != "mjpeg" ] && [ "$FORMAT" != "yuyv422" ]; then
    echo "Invalid format. Use mjpeg or yuyv422."
    exit 1
fi

# Run the capture
echo "Capturing $FRAMES frames from $DEVICE using $FORMAT at $FRAMERATE fps with $RESOLUTION"
ffmpeg -f v4l2 -input_format $FORMAT -r $FRAMERATE -s $RESOLUTION -i $DEVICE -frames:v $FRAMES frame_%06d.jpg
