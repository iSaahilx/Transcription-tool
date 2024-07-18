import os
from pydub import AudioSegment
from pydub.silence import split_on_silence
from openai import OpenAI

# Function to split audio into chunks
def split_audio(input_audio_path, output_dir):
    sound = AudioSegment.from_file(input_audio_path, format="mp3")
    chunks = split_on_silence(sound, min_silence_len=500, silence_thresh=-40)
    for i, chunk in enumerate(chunks):
        chunk.export(os.path.join(output_dir, f"chunk{i}.mp3"), format="mp3")

# Function to transcribe audio chunk
def transcribe_chunk(api_key, chunk_path):
    client = OpenAI(api_key=api_key)
    with open(chunk_path, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            model="whisper-1", 
            file=audio_file
        )
    return transcription.text

# Main function
def main():
    api_key = "YOUR API KEY"
    audio_file_path = "audio_files/audio.mp3"
    output_dir = "/audio_files"

    # Split audio into chunks
    split_audio(audio_file_path, output_dir)

    # Transcribe each chunk
    transcriptions = []
    num_chunks = len(os.listdir(output_dir))
    for i in range(num_chunks):
        chunk_path = os.path.join(output_dir, f"chunk{i}.mp3")

        # Check if chunk file exists
        if os.path.exists(chunk_path):
            transcription_text = transcribe_chunk(api_key, chunk_path)
            transcriptions.append(transcription_text)

            # Delete the chunk file after transcription
            os.remove(chunk_path)
            print(f"Deleted {chunk_path}")
        else:
            print(f"Chunk file {chunk_path} does not exist.")

    # Combine transcriptions
    full_transcription = "\n".join(transcriptions)
    print("\nFull Transcription:")
    print(full_transcription)

if __name__ == "__main__":
    main()
