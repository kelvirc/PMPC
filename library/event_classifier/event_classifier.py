import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from concurrent.futures import ProcessPoolExecutor
import os
import multiprocessing

class EventClassifier:
    def __init__(self, model, tokenizer, metadata_df, transcription_path, logs_folder_path, cores=None):
        """
        Initialize the EventClassifier class.

        :param model: The BERT model to be used for classification.
        :param tokenizer: The tokenizer corresponding to the BERT model.
        :param metadata_df: DataFrame with metadata about the files.
        :param transcription_path: Path to the directory containing transcriptions.
        :param logs_folder_path: Path to the directory where processed log files will be saved.
        :param cores: Number of workers for parallel processing (default is None, uses all available cores).
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.tokenizer = tokenizer
        self.metadata_df = metadata_df
        self.transcription_path = transcription_path
        self.logs_folder_path = logs_folder_path
        self.cores = cores

        self.label_descriptions = {
            "Introduction": "The beginning of the conversation, including greetings and opening statements.",
            "Problem Statement": "A description of a problem or issue.",
            "Resolution": "A solution or course of action to solve the problem.",
            "Call Deposition": "The end of the call or closing statements.",
            "No Voice Detected": "No voice or meaningful text detected in this segment."
        }

        self.classification_data = pd.DataFrame()

    def classify_with_bert(self, text):
        """
        Classify a segment of text using the BERT model.

        :param text: The text to classify.
        :return: Predicted label for the text.
        """
        if not isinstance(text, str):
            raise ValueError("Text input must be of type str.")

        if text.strip() == "":
            return "No Voice Detected"

        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            predicted_class = torch.argmax(logits, dim=-1).item()

        predicted_label = list(self.label_descriptions.keys())[predicted_class]
        return predicted_label

    def process_file(self, file_name):
        """
        Process a single transcription file to classify text segments and save the results as a log file.

        :param file_name: The name of the file to process.
        :return: DataFrame with classification results.
        """
        transcription_file = os.path.join(self.transcription_path, file_name.replace('.mp3', '_transcription.csv'))
        try:
            # Load transcription
            transcription_df = pd.read_csv(transcription_file)

            # Ensure there's a 'Text' column in the transcription file
            if 'Text' not in transcription_df.columns:
                raise ValueError(f"The transcription file {transcription_file} must contain a 'Text' column.")

            # Clean the 'Text' column by ensuring all values are strings
            transcription_df["Text"] = transcription_df["Text"].fillna("").astype(str)

            # Add a column for the source file name
            transcription_df["file_name"] = file_name

            # Apply classification
            transcription_df["flow_label"] = transcription_df["Text"].apply(self.classify_with_bert)

            # Reorder columns to have 'file_name' as the first column
            cols = ["file_name"] + [col for col in transcription_df.columns if col != "file_name"]
            transcription_df = transcription_df[cols]

            # Save log file without the 'file_name' column
            log_file_name = os.path.join(self.logs_folder_path, file_name.replace('.mp3', '_log.csv'))
            transcription_df.drop(columns=["file_name"], inplace=True)
            transcription_df.to_csv(log_file_name, index=False)

            return transcription_df
        except Exception as e:
            print(f"Error processing file {file_name}: {e}")
            return pd.DataFrame()

    def process_all_files(self):
        """
        Process all transcriptions in the metadata DataFrame.

        :return: Consolidated classification results for all files.
        """
        files_to_process = self.metadata_df[self.metadata_df["transcription_available"] == True]["file_name"]

        # Set multiprocessing start method to spawn
        multiprocessing.set_start_method("spawn", force=True)

        with ProcessPoolExecutor(max_workers=self.cores) as executor:
            futures = {executor.submit(self.process_file, file_name): file_name for file_name in files_to_process}

            for future in futures:
                try:
                    result = future.result()
                    if not result.empty:
                        self.classification_data = pd.concat([self.classification_data, result], ignore_index=True)
                except Exception as e:
                    print(f"Error during parallel processing: {e}")

    def get_data(self):
        """
        Retrieve the consolidated classification data.
        """
        return self.classification_data