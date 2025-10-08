# Audio-Video-Visualizer

A Python-based music visualization tool that creates stunning, beat-synchronized visual representations from video files containing audio. Perfect for visualizing speaker outputs, music performances, or any audio content.

## Features

- Extracts audio from video files automatically
- Real-time beat detection and synchronization
- Multiple visualization layers:
  - Circular frequency waveform with color gradients
  - Spectrum analyzer bars
  - Dynamic center pulse responding to beats
- Maintains original video dimensions and frame rate
- Outputs synchronized MP4 file with same duration as input

## Demo

The visualizer creates three synchronized visual elements:
- Circular waveform that responds to frequency content
- Bottom frequency bars showing spectrum analysis
- Center pulsing circle that intensifies on beat drops

## Requirements

- Python 3.7+
- FFmpeg (must be installed on your system)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/audio-video-visualizer.git
cd audio-video-visualizer
```

2. Install required Python packages:
```bash
pip install numpy opencv-python librosa matplotlib scipy
```

3. Install FFmpeg:
   - Ubuntu/Debian: `sudo apt-get install ffmpeg`
   - macOS: `brew install ffmpeg`
   - Windows: Download from [ffmpeg.org](https://ffmpeg.org/download.html)

## Usage

Basic usage:

```python
from music_visualizer import MusicVisualizer

# Create visualizer instance
visualizer = MusicVisualizer(
    video_path="input_video.mp4",
    output_path="output_visualization.mp4"
)

# Process the video
visualizer.process()
```

Command line usage:

```bash
python music_visualizer.py --input your_video.mp4 --output visualization.mp4
```

## How It Works

1. Audio Extraction: Uses FFmpeg to extract audio track from video
2. Audio Analysis: Employs librosa for:
   - Beat detection and tempo calculation
   - Frequency spectrum analysis using mel spectrogram
   - RMS energy calculation for amplitude tracking
3. Visualization Generation: Creates frame-by-frame visuals synchronized to audio features
4. Video Export: Outputs MP4 file with same specifications as input

## Customization

You can customize visualization parameters by modifying these values in the code:

```python
# Number of frequency bars
num_bars = 32

# Number of circular waveform points
num_points = 64

# Beat sensitivity threshold
beat_threshold = 0.1
```

## Example Output

Input: Video of JBL Flip 7 speaker playing music
Output: Synchronized visualization with frequency spectrum, waveforms, and beat pulses

## Technical Details

- Audio Analysis: 44.1kHz sample rate, 512 hop length
- Frequency Analysis: 128 mel bands
- Beat Detection: Dynamic threshold with tempo tracking
- Video Codec: MP4V
- Color Scheme: HSV-based gradient from red (low freq) to blue (high freq)

## Use Cases

- Music production and DJ performances
- Educational demonstrations of audio frequencies
- Social media content creation
- Speaker and audio equipment demonstrations
- Live performance visualizations
- Podcast and audio content enhancement

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## Known Limitations

- Processing time depends on video length (approximately 1:1 ratio)
- Requires sufficient RAM for longer videos
- FFmpeg must be accessible from command line

## Future Enhancements

- [ ] Multiple visualization style presets
- [ ] Real-time visualization option
- [ ] GPU acceleration for faster processing
- [ ] Custom color scheme selection
- [ ] Audio reactivity sensitivity controls
- [ ] Support for audio-only input files
- [ ] Web-based interface

## License

MIT License - feel free to use this project for personal or commercial purposes.

## Acknowledgments

- librosa for audio analysis capabilities
- OpenCV for video processing
- FFmpeg for audio extraction

## Contact

For questions, suggestions, or issues, please open an issue on GitHub.

---

Made with ♪ for music visualization enthusiasts
