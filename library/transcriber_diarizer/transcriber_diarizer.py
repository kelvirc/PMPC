from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import pandas as pd
from pydub import AudioSegment
from resemblyzer import VoiceEncoder, preprocess_wav
from sklearn.cluster import AgglomerativeClustering
import threading


class CallTranscriberDiarizer:
    def __init__(self, audio_path, transcription_path, whisper_model, wav_conversion=True, language="english"):
        """
        Initialize the CallTranscriberDiarizer class.

        Parameters:
        - audio_path (str): Path to the folder containing audio files.
        - transcription_path (str): Path to save transcription results.
        - whisper_model: Preloaded Whisper model.
        - wav_conversion (bool): Whether to convert MP3 to WAV for Resemblyzer (default is True).
        - language (str): Language for transcription.
        """
        self.audio_path = audio_path
        self.transcription_path = transcription_path
        self.model = whisper_model
        self.wav_conversion = wav_conversion
        self.language = language
        self.transcription_data = pd.DataFrame()
        self.diarization_data = pd.DataFrame()
        self.lock = threading.Lock()
        self.encoder = VoiceEncoder()  # Single instance of VoiceEncoder

    def _log(self, message, level="info"):
        """
        Simple logger for displaying messages.
        """
        print(f"[{level.upper()}] {message}")

    def _convert_to_wav(self, audio_file):
        """
        Convert MP3 file to WAV format for Resemblyzer.
        """
        file_name = os.path.splitext(os.path.basename(audio_file))[0]
        processed_folder = os.path.join(self.audio_path, "processed_calls")
        os.makedirs(processed_folder, exist_ok=True)
        wav_path = os.path.join(processed_folder, f"{file_name}.wav")
        try:
            audio = AudioSegment.from_file(audio_file)
            audio.export(wav_path, format="wav")
            return wav_path
        except Exception as e:
            self._log(f"Error converting {audio_file}: {e}", "error")
            return None

    def _transcribe_and_diarize(self, audio_file):
        """
        Transcribe and diarize a single file.
        """
        try:
            file_name = os.path.splitext(os.path.basename(audio_file))[0]
            self._log(f"Starting transcription for {audio_file}", "info")

            # Transcription
            file_size = os.path.getsize(audio_file)
            self._log(f"File size: {file_size} bytes", "debug")

            with self.lock:
                self._log(f"Calling Whisper model.transcribe() for {audio_file}", "debug")
                result = self.model.transcribe(audio_file, word_timestamps=True, language=self.language)

            self._log(f"Raw transcription result keys for {audio_file}: {list(result.keys())}", "debug")

            if "segments" not in result or not result["segments"]:
                self._log(f"No valid segments found for {audio_file}", "error")
                return None

            transcription_df = pd.DataFrame(result["segments"])
            transcription_df = transcription_df[["start", "end", "text"]]
            transcription_df.rename(columns={"start": "Start Time (s)", "end": "End Time (s)", "text": "Text"}, inplace=True)
            transcription_df["file_name"] = file_name

            self._log(f"Successfully formatted DataFrame for {audio_file}", "info")

            # WAV Conversion
            wav_path = None
            if self.wav_conversion:
                wav_path = self._convert_to_wav(audio_file)

            # Diarization
            diarized_df = self._process_diarization(wav_path, transcription_df) if wav_path else None
            if diarized_df is not None:
                # Save enriched transcription
                enriched_df = diarized_df.drop(columns=["file_name"])  # Drop file_name column for CSV
                output_file = os.path.join(self.transcription_path, f"{file_name}_transcription.csv")
                enriched_df.to_csv(output_file, index=False)
                self._log(f"Enriched transcription with speakers saved to {output_file}", "info")

            return diarized_df
        except Exception as e:
            self._log(f"Error transcribing and diarizing {audio_file}: {e}", "error")
            return None

    def process_transcriptions_and_diarizations(self, call_metadata_df, cores=4):
        """
        Transcribe and diarize audio files in parallel.
        Only process files that have transcription_available=False in the metadata.
        """
        audio_files = [
            os.path.join(self.audio_path, f)
            for f in os.listdir(self.audio_path)
            if f.endswith(".mp3")
        ]

        # Filter metadata for files that exist in audio_path and need transcription
        audio_file_names = {os.path.basename(f) for f in audio_files}
        files_to_process = call_metadata_df[
            (call_metadata_df["file_name"].isin(audio_file_names)) & 
            (~call_metadata_df["transcription_available"])
        ]["file_name"].tolist()

        # Build the final list of audio files to process
        audio_files = [os.path.join(self.audio_path, f) for f in files_to_process]

        self._log(f"Processing transcriptions and diarizations for {len(audio_files)} files using {cores} threads.", "info")

        diarization_results = []

        with ThreadPoolExecutor(max_workers=cores) as executor:
            futures = {executor.submit(self._transcribe_and_diarize, audio_file): audio_file for audio_file in audio_files}
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    diarization_results.append(result)

        self.diarization_data = pd.concat(diarization_results, ignore_index=True) if diarization_results else pd.DataFrame()

    def _process_diarization(self, wav_path, transcriptions):
        """
        Process diarization for a single file, align speakers, and return results.
        """
        try:
            file_name = os.path.splitext(os.path.basename(wav_path))[0]

            if not os.path.exists(wav_path):
                self._log(f"Processed WAV file not found for {wav_path}", "error")
                return None

            wav = preprocess_wav(wav_path)

            embeddings, timestamps = self._extract_embeddings(wav)

            num_speakers = 2
            clustering_model = AgglomerativeClustering(n_clusters=num_speakers, metric="euclidean", linkage="ward")
            speaker_labels = clustering_model.fit_predict(embeddings)

            diarized_df = self._assign_speakers_to_transcriptions(transcriptions, timestamps, speaker_labels)
            diarized_df["file_name"] = file_name

            self._log(f"Diarization completed for {file_name}", "info")
            return diarized_df
        except Exception as e:
            self._log(f"Error during diarization for {wav_path}: {e}", "error")
            return None

    def _extract_embeddings(self, wav, window_length_sec=1.5, step_length_sec=0.75):
        """
        Extract speaker embeddings from WAV using Resemblyzer.
        """
        wav_duration = len(wav) / 16000
        embeddings, timestamps = [], []
        step_length_samples = int(step_length_sec * 16000)
        window_length_samples = int(window_length_sec * 16000)

        for start_sample in range(0, len(wav), step_length_samples):
            end_sample = start_sample + window_length_samples
            wav_segment = wav[start_sample:end_sample]
            embedding = self.encoder.embed_utterance(wav_segment)
            embeddings.append(embedding)
            timestamps.append((start_sample / 16000, end_sample / 16000))
        return embeddings, timestamps

    def _assign_speakers_to_transcriptions(self, transcriptions, speaker_timestamps, speaker_labels):
        """
        Align transcriptions with speaker labels.
        """
        transcription_speakers = []
        for _, row in transcriptions.iterrows():
            segment_start, segment_end = row["Start Time (s)"], row["End Time (s)"]
            segment_speakers = [
                label for (start_time, end_time), label in zip(speaker_timestamps, speaker_labels)
                if start_time < segment_end and end_time > segment_start
            ]
            majority_speaker = max(set(segment_speakers), key=segment_speakers.count) if segment_speakers else -1
            transcription_speakers.append({
                "Start Time (s)": segment_start,
                "End Time (s)": segment_end,
                "Text": row["Text"],
                "Speaker": f"Speaker {majority_speaker}" if majority_speaker != -1 else "Unknown"
            })
        return pd.DataFrame(transcription_speakers)

    def get_data(self):
        """
        Retrieve the consolidated diarization data.
        """
        return self.diarization_data
