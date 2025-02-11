import os
import openai
import shutil
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings


class AvalaiClient:
    def __init__(self, api_key="", base_url="", model_name="gpt-4o-mini"):
        self.api_key, self.base_url = self._set_api_config(api_key, base_url)
        self.model_name = model_name
        self.client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url)

    def get_chat_model(self):
        return ChatOpenAI(model_name=self.model_name, openai_api_key=self.api_key, openai_api_base=self.base_url)

    def get_embeddings(self):
        return OpenAIEmbeddings(openai_api_key=self.api_key, openai_api_base=self.base_url)

    def get_audio_model(self):
        return self.client.audio

    def _set_api_config(self, api_key, base_url):
        default_api_key = "aa-UGqhFeHZm86jdMFDu1DoJ2q2OgThodQlHuCfhu7OBCPXMruY"
        default_base_url = "https://api.avalai.ir/v1"
        return api_key or default_api_key, base_url or default_base_url


avalai_client = AvalaiClient()
llm = avalai_client.get_chat_model()


def trim_audio(input_file, output_file, start=0, duration=30):
    """ Trims an audio file to the first 30 seconds to speed up processing. """
    os.system(f'ffmpeg -y -i "{input_file}" -ss {start} -t {duration} -c copy "{output_file}"')


def transcribe_audio(audio_file):
    """ Transcribes the trimmed audio file using Whisper. """
    with open(audio_file, "rb") as file:
        transcript = avalai_client.get_audio_model().transcriptions.create(model="whisper-1", file=file)
    return transcript.text


def identify_audiobook(text):
    """ Identifies the audiobook title from the transcribed text. """
    prompt = f"""The following is an excerpt from an audiobook. 
    Identify the **exact title of the book** and return **only the title**, nothing else.
    If the title contains extra descriptive words, return only the **official book title**:\n\n{text}"""

    response = llm.predict(prompt)
    return response.strip().split(" نوشته")[0]  # Extract only the title


def process_audiobooks(folder_path):
    """ Processes all MP3 files in the given folder, renames them with their correct book title. """
    for file in os.listdir(folder_path):
        if file.endswith(".mp3"):
            input_path = os.path.join(folder_path, file)
            output_trimmed = os.path.join(folder_path, "temp_trimmed.mp3")

            print(f"\n🔄 Processing: {file}")

            try:
                # Step 1: Trim the audio
                trim_audio(input_path, output_trimmed)

                # Step 2: Transcribe the audio
                transcript_text = transcribe_audio(output_trimmed)

                # Step 3: Identify the audiobook title
                audiobook_name = identify_audiobook(transcript_text)

                # Step 4: Rename the file
                new_filename = f"{audiobook_name}.mp3"
                new_path = os.path.join(folder_path, new_filename)

                # Avoid overwriting existing files
                if os.path.exists(new_path):
                    new_path = os.path.join(folder_path, f"{audiobook_name}_{file}")

                shutil.move(input_path, new_path)
                print(f"✅ Renamed to: {new_filename}")

            except Exception as e:
                print(f"❌ Error processing {file}: {e}")

            finally:
                # Clean up temp files
                if os.path.exists(output_trimmed):
                    os.remove(output_trimmed)


# 📂 Get folder path from user
folder_path = input("Enter the folder path containing audiobooks: ").strip()

if os.path.isdir(folder_path):
    process_audiobooks(folder_path)
else:
    print("❌ Invalid folder path. Please enter a valid path.")
