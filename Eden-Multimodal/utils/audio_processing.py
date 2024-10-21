import torch
import librosa
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from config import whisper_model_name

whisper_processor = WhisperProcessor.from_pretrained(whisper_model_name)
whisper_model = WhisperForConditionalGeneration.from_pretrained(whisper_model_name)

def transcribe_speech(audiopath):
    speech, rate = librosa.load(audiopath, sr=16000)
    audio_input = whisper_processor(speech, return_tensors="pt", sampling_rate=16000)
    
    with torch.no_grad():
        generated_ids = whisper_model.generate(audio_input["input_features"])
    
    transcription = whisper_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    return transcription

def getAudioArray(audio_path):
    speech, rate = librosa.load(audio_path, sr=16000)
    return speech