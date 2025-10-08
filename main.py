import numpy as np
import cv2
import librosa
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.signal import butter, filtfilt
import subprocess
import os

class MusicVisualizer:
    def __init__(self, video_path, output_path='output_visualization.mp4'):
        self.video_path = video_path
        self.output_path = output_path
        self.temp_audio = 'temp_audio.wav'
        
    def extract_audio(self):
        """Extract audio from video file"""
        cmd = [
            'ffmpeg', '-i', self.video_path,
            '-vn', '-acodec', 'pcm_s16le',
            '-ar', '44100', '-ac', '2',
            self.temp_audio, '-y'
        ]
        subprocess.run(cmd, capture_output=True)
        print("Audio extracted successfully")
        
    def analyze_audio(self):
        """Analyze audio for visualization parameters"""
        y, sr = librosa.load(self.temp_audio, sr=None)
        self.sample_rate = sr
        self.duration = librosa.get_duration(y=y, sr=sr)
        
        # Calculate tempo and beat frames
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        self.tempo = float(tempo) if isinstance(tempo, np.ndarray) else tempo
        self.beat_times = librosa.frames_to_time(beat_frames, sr=sr)
        
        # Calculate spectral features
        hop_length = 512
        self.stft = np.abs(librosa.stft(y, hop_length=hop_length))
        self.times = librosa.frames_to_time(np.arange(self.stft.shape[1]), 
                                           sr=sr, hop_length=hop_length)
        
        # Mel spectrogram for frequency bands
        self.mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        self.mel_spec_db = librosa.power_to_db(self.mel_spec, ref=np.max)
        
        # RMS energy for amplitude
        self.rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
        self.rms = (self.rms - self.rms.min()) / (self.rms.max() - self.rms.min())
        
        print(f"Audio analyzed: {self.duration:.2f}s, tempo: {self.tempo:.1f} BPM")
        
    def create_visualization(self):
        """Create visualization video"""
        # Get video properties
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.output_path, fourcc, fps, (width, height))
        
        total_frames = int(self.duration * fps)
        
        print(f"Creating visualization: {width}x{height} @ {fps} fps")
        
        for frame_idx in range(total_frames):
            current_time = frame_idx / fps
            
            # Find closest audio analysis frame
            time_idx = np.argmin(np.abs(self.times - current_time))
            
            # Create blank frame
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Check if current time is near a beat
            beat_intensity = 0
            for beat_time in self.beat_times:
                if abs(current_time - beat_time) < 0.1:
                    beat_intensity = max(0, 1 - abs(current_time - beat_time) * 10)
            
            # Get current audio features
            current_rms = self.rms[time_idx] if time_idx < len(self.rms) else 0
            
            # Visualization 1: Circular waveform
            self.draw_circular_waveform(frame, width, height, time_idx, 
                                       current_rms, beat_intensity)
            
            # Visualization 2: Frequency bars
            self.draw_frequency_bars(frame, width, height, time_idx, 
                                    current_rms, beat_intensity)
            
            # Visualization 3: Center pulse
            self.draw_center_pulse(frame, width, height, current_rms, beat_intensity)
            
            # Add time indicator
            time_text = f"Time: {current_time:.2f}s"
            cv2.putText(frame, time_text, (20, height - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            out.write(frame)
            
            if frame_idx % 30 == 0:
                print(f"Progress: {frame_idx}/{total_frames} frames ({frame_idx/total_frames*100:.1f}%)")
        
        out.release()
        print(f"Visualization saved to {self.output_path}")
        
    def draw_circular_waveform(self, frame, width, height, time_idx, rms, beat):
        """Draw circular waveform visualization"""
        center_x, center_y = width // 2, height // 2
        base_radius = min(width, height) // 4
        
        num_points = 64
        spectrum_slice = self.mel_spec_db[:num_points, time_idx] if time_idx < self.mel_spec_db.shape[1] else np.zeros(num_points)
        normalized = (spectrum_slice - spectrum_slice.min()) / (spectrum_slice.max() - spectrum_slice.min() + 1e-6)
        
        for i in range(num_points):
            angle1 = (i / num_points) * 2 * np.pi
            angle2 = ((i + 1) / num_points) * 2 * np.pi
            
            radius = base_radius + normalized[i] * 100 * (1 + beat * 0.3)
            
            x1 = int(center_x + radius * np.cos(angle1))
            y1 = int(center_y + radius * np.sin(angle1))
            x2 = int(center_x + radius * np.cos(angle2))
            y2 = int(center_y + radius * np.sin(angle2))
            
            # Color based on frequency (low = red, high = blue)
            hue = int(180 * i / num_points)
            color = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0]
            color = tuple(map(int, color))
            
            cv2.line(frame, (x1, y1), (x2, y2), color, 2)
    
    def draw_frequency_bars(self, frame, width, height, time_idx, rms, beat):
        """Draw frequency spectrum bars"""
        num_bars = 32
        bar_width = width // (num_bars + 1)
        
        spectrum_slice = self.mel_spec_db[:num_bars, time_idx] if time_idx < self.mel_spec_db.shape[1] else np.zeros(num_bars)
        normalized = (spectrum_slice - spectrum_slice.min()) / (spectrum_slice.max() - spectrum_slice.min() + 1e-6)
        
        for i in range(num_bars):
            bar_height = int(normalized[i] * height * 0.3 * (1 + beat * 0.2))
            x = (i + 1) * bar_width
            y_start = height - 50
            y_end = y_start - bar_height
            
            # Gradient color
            hue = int(180 * i / num_bars)
            color = cv2.cvtColor(np.uint8([[[hue, 200, 255]]]), cv2.COLOR_HSV2BGR)[0][0]
            color = tuple(map(int, color))
            
            cv2.rectangle(frame, (x - bar_width//3, y_start), 
                         (x + bar_width//3, y_end), color, -1)
    
    def draw_center_pulse(self, frame, width, height, rms, beat):
        """Draw pulsing center circle"""
        center_x, center_y = width // 2, height // 2
        radius = int(50 + rms * 30 + beat * 40)
        
        # Outer glow
        cv2.circle(frame, (center_x, center_y), radius + 10, (50, 100, 200), 2)
        cv2.circle(frame, (center_x, center_y), radius, (100, 200, 255), -1)
        
        # Beat flash
        if beat > 0.5:
            cv2.circle(frame, (center_x, center_y), radius + 20, 
                      (255, 255, 255), 3)
    
    def process(self):
        """Main processing pipeline"""
        print("Starting music visualization process...")
        self.extract_audio()
        self.analyze_audio()
        self.create_visualization()
        
        # Cleanup
        if os.path.exists(self.temp_audio):
            os.remove(self.temp_audio)
        
        print("Process complete!")

# Usage example
if __name__ == "__main__":
    # Replace with your video path
    input_video = "/content/vibration_mechanism_in_flip_7 (720p).mp4"
    output_video = "music_visualization.mp4"
    
    visualizer = MusicVisualizer(input_video, output_video)
    visualizer.process()
