# sudo apt-get install -y libasound-dev portaudio19-dev libportaudio2 libportaudiocpp0 python3-dev ffmpeg
# For windows installing ffmpeg is different. Cannot use apt install
# pip install pyaudio wave openai-whisper ollama

import pyaudio
import wave
import whisper
import ollama
import time


def record_audio(filename="recording.wav", duration=5):
    # Audio parameters
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000

    # Initialize PyAudio
    p = pyaudio.PyAudio()

    # Open stream
    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK
    )

    print("* Recording...")
    frames = []

    # Record for specified duration
    for i in range(0, int(RATE / CHUNK * duration)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("* Done recording")

    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    p.terminate()

    # Save the recorded data as a WAV file
    wf = wave.open(filename, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    return filename


def transcribe_audio(filename):
    # Load Whisper model
    model = whisper.load_model("base")

    # Transcribe audio
    result = model.transcribe(filename)

    # Get transcription text
    return result["text"]


def summarize_text(text):
    # Use Ollama to summarize the text
    response = ollama.chat(model='llama3.2:1b', messages=[
        {
            'role': 'user',
            'content': f'Please summarize this text concisely: {text}'
        }
    ])
    return response['message']['content']


def main():
    # Record audio
    audio_file = record_audio(duration=10)  # Records for 10 seconds

    # Transcribe audio
    print("\nTranscribing audio...")
    transcription = transcribe_audio(audio_file)
    print("\nTranscription:", transcription)

    # Summarize transcription
    print("\nGenerating summary...")
    summary = summarize_text(transcription)
    print("\nSummary:", summary)


if __name__ == "__main__":
    main()
