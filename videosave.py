#!/usr/bin/env python3
import sys
import time
import threading
import wave
import os
from collections import deque
from datetime import datetime

import numpy as np
import cv2
import mss
import pyaudio
import speech_recognition as sr
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QLabel,
    QComboBox, QTextEdit, QHBoxLayout, QSlider
)
from PyQt6.QtCore import pyqtSignal, QObject, Qt

import subprocess
import simpleaudio as sa  # For notification sound playback

# === PATH SETUP ===
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "clips")
NOTIFY_WAV = os.path.join(SCRIPT_DIR, "notify.wav")
os.makedirs(OUTPUT_DIR, exist_ok=True)
# ==================

# ==== Settings ====
FPS = 20
BUFFER_SECONDS = 30
# ==================

class Logger(QObject):
    new_log = pyqtSignal(str)

class ClipperApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Garmin Clipper")

        self.logger = Logger()
        self.logger.new_log.connect(self.append_log)

        layout = QVBoxLayout()

        # Monitor select
        self.monitor_label = QLabel("Select Monitor:")
        self.monitor_combo = QComboBox()
        layout.addWidget(self.monitor_label)
        layout.addWidget(self.monitor_combo)

        # Mic select
        self.mic_label = QLabel("Select Microphone:")
        self.mic_combo = QComboBox()
        layout.addWidget(self.mic_label)
        layout.addWidget(self.mic_combo)

        # Mic volume slider
        self.mic_vol_label = QLabel("Mic Volume: 100%")
        self.mic_vol_slider = QSlider(Qt.Orientation.Horizontal)
        self.mic_vol_slider.setMinimum(0)
        self.mic_vol_slider.setMaximum(200)
        self.mic_vol_slider.setValue(100)
        self.mic_vol_slider.setTickInterval(10)
        self.mic_vol_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.mic_vol_slider.valueChanged.connect(self.update_mic_volume_label)
        layout.addWidget(self.mic_vol_label)
        layout.addWidget(self.mic_vol_slider)

        # Desktop audio volume slider
        self.desktop_vol_label = QLabel("Desktop Audio Volume: 100%")
        self.desktop_vol_slider = QSlider(Qt.Orientation.Horizontal)
        self.desktop_vol_slider.setMinimum(0)
        self.desktop_vol_slider.setMaximum(200)
        self.desktop_vol_slider.setValue(100)
        self.desktop_vol_slider.setTickInterval(10)
        self.desktop_vol_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.desktop_vol_slider.valueChanged.connect(self.update_desktop_volume_label)
        layout.addWidget(self.desktop_vol_label)
        layout.addWidget(self.desktop_vol_slider)

        # Start/Stop buttons
        buttons_layout = QHBoxLayout()
        self.start_btn = QPushButton("Start")
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        buttons_layout.addWidget(self.start_btn)
        buttons_layout.addWidget(self.stop_btn)
        layout.addLayout(buttons_layout)

        # Manual test save button
        self.test_save_btn = QPushButton("Test Save Clip Now")
        self.test_save_btn.setEnabled(False)
        layout.addWidget(self.test_save_btn)

        # Log output
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        layout.addWidget(self.log_output)

        self.setLayout(layout)

        self.start_btn.clicked.connect(self.start_recording)
        self.stop_btn.clicked.connect(self.stop_recording)
        self.test_save_btn.clicked.connect(lambda: threading.Thread(target=self.save_clip, daemon=True).start())

        # PyAudio and mss init
        self.pyaudio_instance = pyaudio.PyAudio()
        self.audio_stream = None

        # video frames buffer (deque of frames)
        self.frame_buffer = deque(maxlen=BUFFER_SECONDS * FPS)

        # audio chunks buffer (deque of raw bytes). We'll manage byte length ourselves.
        self.audio_chunks = deque()
        self.audio_bytes_len = 0
        self.audio_lock = threading.Lock()

        # parameters set at start_recording
        self.sample_rate = None
        self.sample_width = None
        self.channels = 1
        self.audio_max_bytes = None  # to be computed

        self.recording = False
        self.video_thread = None
        self.listen_thread = None
        self.stop_flag = threading.Event()

        self.populate_devices()

    def update_mic_volume_label(self):
        val = self.mic_vol_slider.value()
        self.mic_vol_label.setText(f"Mic Volume: {val}%")

    def update_desktop_volume_label(self):
        val = self.desktop_vol_slider.value()
        self.desktop_vol_label.setText(f"Desktop Audio Volume: {val}%")

    def log(self, msg):
        self.logger.new_log.emit(f"{datetime.now().strftime('%H:%M:%S')} - {msg}")

    def append_log(self, msg):
        self.log_output.append(msg)

    def populate_devices(self):
        # List monitors using mss
        try:
            with mss.mss() as sct:
                self.monitors = sct.monitors
        except Exception as e:
            self.monitors = [{},]
            self.log(f"mss error listing monitors: {e}")

        self.monitor_combo.clear()
        for i, mon in enumerate(self.monitors):
            if i == 0:
                self.monitor_combo.addItem("All monitors (virtual screen)", i)
            else:
                self.monitor_combo.addItem(
                    f"Monitor {i}: {mon['width']}x{mon['height']} @ ({mon['left']},{mon['top']})", i
                )

        # List mic devices PyAudio
        self.mic_combo.clear()
        try:
            for i in range(self.pyaudio_instance.get_device_count()):
                dev = self.pyaudio_instance.get_device_info_by_index(i)
                if dev.get('maxInputChannels', 0) > 0:
                    self.mic_combo.addItem(dev.get('name'), i)
        except Exception as e:
            self.log(f"PyAudio device enumeration error: {e}")

    def start_recording(self):
        if self.recording:
            return
        if self.monitor_combo.count() == 0:
            self.log("No monitors available.")
            return
        if self.mic_combo.count() == 0:
            self.log("No microphone devices available.")
            return

        self.recording = True
        self.stop_flag.clear()
        self.frame_buffer.clear()
        with self.audio_lock:
            self.audio_chunks.clear()
            self.audio_bytes_len = 0

        self.log("Starting recording...")

        self.selected_monitor_index = self.monitor_combo.currentData()
        self.selected_mic_index = self.mic_combo.currentData()

        try:
            mic_info = self.pyaudio_instance.get_device_info_by_index(self.selected_mic_index)
            self.sample_rate = int(mic_info.get('defaultSampleRate', 44100))
            self.sample_width = self.pyaudio_instance.get_sample_size(pyaudio.paInt16)
            self.channels = 1
            self.audio_max_bytes = BUFFER_SECONDS * self.sample_rate * self.sample_width * self.channels

            self.audio_stream = self.pyaudio_instance.open(
                format=pyaudio.paInt16,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=1024,
                input_device_index=self.selected_mic_index,
                stream_callback=self.audio_callback
            )
            self.audio_stream.start_stream()
            self.log(f"Audio stream started: rate={self.sample_rate}, width={self.sample_width}")
        except Exception as e:
            self.log(f"Failed to open audio stream: {e}")
            self.recording = False
            return

        self.video_thread = threading.Thread(target=self.capture_screen_loop, daemon=True)
        self.listen_thread = threading.Thread(target=self.listen_for_triggers, daemon=True)

        self.video_thread.start()
        self.listen_thread.start()

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.test_save_btn.setEnabled(True)

    def stop_recording(self):
        if not self.recording:
            return
        self.recording = False
        self.stop_flag.set()

        try:
            if self.audio_stream is not None:
                self.audio_stream.stop_stream()
                self.audio_stream.close()
                self.audio_stream = None
        except Exception as e:
            self.log(f"Error stopping audio stream: {e}")

        self.log("Stopped recording.")
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.test_save_btn.setEnabled(False)

    def audio_callback(self, in_data, frame_count, time_info, status):
        if not self.recording:
            return (None, pyaudio.paContinue)
        chunk = in_data
        with self.audio_lock:
            self.audio_chunks.append(chunk)
            self.audio_bytes_len += len(chunk)
            while self.audio_bytes_len > self.audio_max_bytes and len(self.audio_chunks) > 0:
                old = self.audio_chunks.popleft()
                self.audio_bytes_len -= len(old)
        return (None, pyaudio.paContinue)

    def capture_screen_loop(self):
        try:
            with mss.mss() as sct:
                monitor = self.monitors[self.selected_monitor_index]
                while not self.stop_flag.is_set():
                    img = np.array(sct.grab(monitor))
                    frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                    self.frame_buffer.append(frame)
                    time.sleep(1 / FPS)
        except Exception as e:
            self.log(f"Screen capture error: {e}")

    def play_notify_sound(self):
        try:
            # Read notify wav raw samples
            with wave.open(NOTIFY_WAV, 'rb') as wf:
                frames = wf.readframes(wf.getnframes())
                sample_width = wf.getsampwidth()
                channels = wf.getnchannels()
                framerate = wf.getframerate()

            # Convert bytes to numpy array
            notify_audio = np.frombuffer(frames, dtype=np.int16)

            # Adjust volume by slider %
            vol_factor = self.desktop_vol_slider.value() / 100.0
            if vol_factor < 0: vol_factor = 0
            # Amplify and clip
            notify_audio = np.clip(notify_audio.astype(np.float32) * vol_factor, -32768, 32767).astype(np.int16)

            # Play with simpleaudio
            wave_obj = sa.WaveObject(notify_audio.tobytes(), channels, sample_width, framerate)
            play_obj = wave_obj.play()
        except Exception as e:
            self.log(f"Failed to play notification sound: {e}")

    def play_notify_sound_blocking(self):
        try:
            # Read notify wav raw samples
            with wave.open(NOTIFY_WAV, 'rb') as wf:
                frames = wf.readframes(wf.getnframes())
                sample_width = wf.getsampwidth()
                channels = wf.getnchannels()
                framerate = wf.getframerate()

            # Convert bytes to numpy array
            notify_audio = np.frombuffer(frames, dtype=np.int16)

            # Adjust volume by slider %
            vol_factor = self.desktop_vol_slider.value() / 100.0
            if vol_factor < 0: vol_factor = 0
            # Amplify and clip
            notify_audio = np.clip(notify_audio.astype(np.float32) * vol_factor, -32768, 32767).astype(np.int16)

            # Play with simpleaudio blocking
            wave_obj = sa.WaveObject(notify_audio.tobytes(), channels, sample_width, framerate)
            play_obj = wave_obj.play()
            play_obj.wait_done()
        except Exception as e:
            self.log(f"Failed to play notification sound: {e}")

    def listen_for_triggers(self):
        recognizer = sr.Recognizer()
        try:
            mic = sr.Microphone(device_index=self.selected_mic_index)
        except Exception as e:
            self.log(f"Could not open Microphone for speech recognition: {e}")
            return

        with mic as source:
            try:
                recognizer.adjust_for_ambient_noise(source, duration=0.5)
            except Exception:
                pass
            self.log("Listening for triggers...")
            triggers = ["okay garmin", "video speichern", "save video", "okay carmen", "okay google", "ok google", "ok carmen", "ok garmin"]
            while not self.stop_flag.is_set():
                try:
                    audio = recognizer.listen(source, timeout=3, phrase_time_limit=5)
                    text = recognizer.recognize_google(audio).lower()
                    self.log(f"Heard: {text}")
                    if any(phrase in text for phrase in triggers):
                        self.log("Trigger phrase detected!")
                        self.play_notify_sound_blocking()
                        threading.Thread(target=self.save_clip, daemon=True).start()
                except sr.WaitTimeoutError:
                    pass
                except sr.UnknownValueError:
                    pass
                except sr.RequestError as e:
                    self.log(f"Speech recognition error: {e}")
                except Exception as e:
                    self.log(f"Listener error: {e}")

    def save_clip(self):
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        abs_output_dir = OUTPUT_DIR
        if not os.path.exists(abs_output_dir):
            os.makedirs(abs_output_dir, exist_ok=True)
            self.log(f"Created clips directory at {abs_output_dir}")

        self.log(f"Saving files to absolute path: {abs_output_dir}")

        video_path = os.path.join(abs_output_dir, f"clip_{ts}.avi")
        audio_path = os.path.join(abs_output_dir, f"clip_{ts}.wav")
        output_path = os.path.join(abs_output_dir, f"clip_{ts}.mp4")

        frames_copy = list(self.frame_buffer)

        with self.audio_lock:
            audio_bytes = b''.join(list(self.audio_chunks))
        if self.audio_max_bytes and len(audio_bytes) > self.audio_max_bytes:
            audio_bytes = audio_bytes[-self.audio_max_bytes:]

        if len(frames_copy) == 0:
            self.log("No video frames in buffer; not saving.")
            return
        if len(audio_bytes) == 0:
            self.log("No audio in buffer; not saving.")
            return

        # Convert captured audio bytes to int16 numpy array
        captured_audio = np.frombuffer(audio_bytes, dtype=np.int16)

        # Apply mic volume slider here before mixing
        mic_vol_factor = self.mic_vol_slider.value() / 100.0
        if mic_vol_factor < 0:
            mic_vol_factor = 0
        captured_audio = np.clip(captured_audio.astype(np.float32) * mic_vol_factor, -32768, 32767).astype(np.int16)

        try:
            with wave.open(NOTIFY_WAV, 'rb') as notif_wav:
                notif_frames = notif_wav.readframes(notif_wav.getnframes())
                notif_rate = notif_wav.getframerate()
                notif_width = notif_wav.getsampwidth()
                notif_channels = notif_wav.getnchannels()

            if (notif_rate == self.sample_rate and
                notif_width == self.sample_width and
                notif_channels == self.channels):

                notif_audio = np.frombuffer(notif_frames, dtype=np.int16)

                # Amplify notify audio before mixing using desktop volume slider
                desktop_vol_factor = self.desktop_vol_slider.value() / 100.0
                if desktop_vol_factor < 0:
                    desktop_vol_factor = 0
                amplify_factor = 30.0 * desktop_vol_factor  # keep previous amplification + slider
                amplified_notify_audio = notif_audio.astype(np.float32) * amplify_factor
                amplified_notify_audio = np.clip(amplified_notify_audio, -32768, 32767).astype(np.int16)

                # Mix notify audio into captured audio at the end
                if len(amplified_notify_audio) > len(captured_audio):
                    captured_audio = np.pad(captured_audio, (0, len(amplified_notify_audio) - len(captured_audio)), mode='constant')

                start_idx = len(captured_audio) - len(amplified_notify_audio)
                mixed_audio = np.copy(captured_audio)

                mixed_audio[start_idx:] = np.clip(
                    mixed_audio[start_idx:].astype(np.int32) + amplified_notify_audio.astype(np.int32),
                    -32768, 32767
                ).astype(np.int16)

                audio_bytes = mixed_audio.tobytes()
                self.log("Mixed amplified notification audio into clip audio.")
            else:
                self.log("Notification audio format mismatch, skipping mixing.")
        except Exception as e:
            self.log(f"Error mixing notification audio: {e}")

        self.log(f"Saving clip with {len(frames_copy)} frames and {len(audio_bytes)} bytes of audio.")

        try:
            h, w, _ = frames_copy[0].shape
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            out = cv2.VideoWriter(video_path, fourcc, FPS, (w, h))
            for frame in frames_copy:
                out.write(frame)
            out.release()
            self.log(f"Video saved temporarily at: {video_path}")
        except Exception as e:
            self.log(f"Error saving video: {e}")
            return

        try:
            wf = wave.open(audio_path, 'wb')
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.sample_width)
            wf.setframerate(self.sample_rate)
            wf.writeframes(audio_bytes)
            wf.close()
            self.log(f"Audio saved temporarily at: {audio_path}")
        except Exception as e:
            self.log(f"Error saving audio: {e}")
            return

        try:
            subprocess.run(['ffmpeg', '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        except Exception as e:
            self.log(f"ffmpeg not found or error running ffmpeg: {e}")
            return

        cmd = [
            'ffmpeg',
            '-y',
            '-i', video_path,
            '-i', audio_path,
            '-c:v', 'libx264',
            '-preset', 'fast',
            '-crf', '23',
            '-c:a', 'aac',
            '-b:a', '192k',
            output_path
        ]
        self.log(f"Running ffmpeg mux command to produce: {output_path}")
        try:
            proc = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            self.log(f"ffmpeg stdout: {proc.stdout.decode(errors='ignore')[:500]}")
            self.log(f"ffmpeg stderr: {proc.stderr.decode(errors='ignore')[:500]}")
            self.log(f"Clip muxed and saved successfully: {output_path}")
        except subprocess.CalledProcessError as e:
            stderr = e.stderr.decode(errors='ignore') if e.stderr else str(e)
            self.log(f"ffmpeg mux error: {stderr[:4000]}")
            return
        except Exception as e:
            self.log(f"ffmpeg mux exception: {e}")
            return

        try:
            os.remove(video_path)
            os.remove(audio_path)
            self.log("Temporary video and audio files removed.")
        except Exception as e:
            self.log(f"Error cleaning up temp files: {e}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = ClipperApp()
    win.show()
    sys.exit(app.exec())
