import pyaudio
import numpy as np
import matplotlib.pyplot as plt

# Configuration
record_seconds = 5  # Adjust the duration of recording as needed

# Initialize the audio stream
audio = pyaudio.PyAudio()

# Open a microphone stream
stream = audio.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=44100,
                    input=True,
                    frames_per_buffer=1024)

print("Recording...")

# Create a figure for plotting
plt.ion()  # Turn on interactive mode for live updating plot
fig, (ax1, ax2) = plt.subplots(2, figsize=(10, 6))
x = np.arange(0, 1024)  # Corrected x array with the correct length
line, = ax1.plot(x, np.random.rand(1024), '-', lw=2)  # Initialize y with the correct length
ax1.set_xlim(0, 1024)
ax1.set_ylim(-32768, 32767)
ax1.set_title('Audio Waveform')
ax1.set_xlabel('Time (samples)')
ax1.set_ylabel('Amplitude')
ax1.grid(True)

xf = np.fft.fftfreq(1024, 1.0 / 44100)
line2, = ax2.semilogx(xf, np.random.rand(1024), '-', lw=2)  # Initialize y with the correct length
ax2.set_xlim(20, 20000)
ax2.set_ylim(0, 50)
ax2.set_title('Frequency Spectrum')
ax2.set_xlabel('Frequency (Hz)')
ax2.set_ylabel('Magnitude (dB)')
ax2.grid(True)

plt.tight_layout()

frames = []

try:
    for _ in range(0, int(44100 / 1024 * record_seconds)):
        data = stream.read(1024)
        frames.append(data)

        # Update the waveform plot
        signal = np.frombuffer(data, dtype=np.int16)
        line.set_ydata(signal)
        plt.draw()

        # Calculate and update the frequency spectrum plot
        spectrum = np.fft.fft(signal)
        magnitude = np.abs(spectrum)
        magnitude_db = 20 * np.log10(magnitude)
        line2.set_ydata(magnitude_db)
        plt.draw()

        # Calculate and print the dominant frequency
        dominant_frequency = xf[np.argmax(magnitude)]
        print(f"Dominant Frequency: {dominant_frequency} Hz")

except KeyboardInterrupt:
    pass

print("Recording finished.")

# Close the audio stream
stream.stop_stream()
stream.close()
audio.terminate()

# Stop the live updating plot
plt.ioff()
plt.show()
