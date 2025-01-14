import os  # For file path manipulation
import pandas as pd  # For working with DataFrames
import torch  # For handling PyTorch tensors
from concurrent.futures import ProcessPoolExecutor  # For parallel processing
import logging  # For logging events and errors
import time  # For tracking performance

class LlamaEventClassifier:
    """
    A standalone classifier tailored for LLaMA models.
    Handles token-level logits and adapts them for classification tasks.
    """
    def __init__(self, model, tokenizer, metadata_df, transcription_path, logs_folder_path, cores=None, device="cpu", context_length=None):
        """
        Initialize the LlamaEventClassifier class.
        :param model: The LLaMA model to be used for classification.
        :param tokenizer: The tokenizer corresponding to the LLaMA model.
        :param metadata_df: DataFrame with metadata about the files.
        :param transcription_path: Path to the directory containing transcriptions.
        :param logs_folder_path: Path to the directory where processed log files will be saved.
        :param cores: Number of workers for parallel processing (default is None, uses all available cores).
        :param context_length: Number of previous and future lines to include in the prompt.
        """
        self.device = device
        self.model = model.to(self.device)
        self.tokenizer = tokenizer
        self.metadata_df = metadata_df
        self.transcription_path = transcription_path
        self.logs_folder_path = logs_folder_path
        self.cores = cores
        self.context_length = context_length

        self.classification_data = pd.DataFrame()
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler("llama_event_classifier.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info("LlamaEventClassifier initialized.")

    def classify_line(self, prompt, max_new_tokens=20, temperature=0.5, top_p=0.5):
        """
        Generate a classification for a given prompt using the Llama-3.2-3B-Instruct model.

        Parameters:
        - prompt (str): The input prompt for the model.
        - max_new_tokens (int): Maximum number of tokens to generate.
        - temperature (float): Sampling temperature for randomness.
        - top_p (float): Top-p sampling for nucleus sampling.

        Returns:
        - str: The model's classification.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(self.device)
        outputs = self.model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=self.tokenizer.pad_token_id
        )
        generated_tokens = outputs[0][inputs.input_ids.shape[1]:]
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return response.strip()

    def build_prompt(self, current_index, df, context_length=None):
        """
        Build a Llama-style prompt with role clarification for classifying 911 call segments.

        Parameters:
        - current_index (int): The index of the current line in the DataFrame.
        - df (pd.DataFrame): The DataFrame containing the transcription data.
        - context_length (int, optional): Number of previous and future lines to include. Default is None (use full context).

        Returns:
        - str: The formatted prompt in the Llama conversation template style.
        """
        total_segments = len(df)
        segment_text = df.loc[current_index, "Text"]
        segment_info = f"Segment {current_index + 1} of {total_segments}: {segment_text}"

        if current_index == 0:  # Special case for call introduction
            start_index = 0
            end_index = 1
        elif context_length is not None:
            start_index = max(0, current_index - context_length)
            end_index = min(len(df), current_index + context_length + 1)
        else:
            start_index = 0
            end_index = len(df)

        context = []
        for i in range(start_index, end_index):
            text = df.loc[i, "Text"]
            marker = "Current" if i == current_index else f"Previous-{current_index - i}" if i < current_index else f"Future+{i - current_index}"
            context.append(f"{marker}: \"{text}\"")

        context_str = "\n".join(context)

        prompt = f"""
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a robot classifying 911 call segments into predefined categories. 
These segments do not have annotations for which speaker, so try to detect which one is the dispatcher and the caller.
Respond with only one category name from the list below. Do not assign multiple categories to a single segment, and without adding any extra words or phrases:

- Call Introduction: 99% of the time happens in the first three segments. The standard phrase usually contains "where is your emergency" and "911".
- Description:  Contains questions and answers related to "what","where" and "who" are involved in the emergency.It comes after the Call Introduction but before Resolution.
- Resolution: Contains a solution to the emergency. Usually contains mentions to police officers, firemen, an ambulance are on the way to the emergency site. It comes after the description and before Call Deposition.
- Call Deposition: Contains closing remarks of "gratitude" and "good bye" words. It is usually at the last segments.

Here are some examples for Call Introduction:

Segment 1 of 20: "Hover County 911, where is your emergency?".
Classification: Call Introduction

Segment 2 of 50: "911, where is your emergency?".
Classification: Call Introduction

Segment 6 of 50: "Where is that?".
Classification: Description

Segment 3 of 30: "Do you know the address?".
Classification: Description

<|start_header_id|>user<|end_header_id|>
Classify this 911 call segment:
{segment_info}
Context:
{context_str}

Your response:<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
        return prompt.strip()

    def classify_with_llama(self, text, current_index, df, context_length=None):
        """
        Classify a segment of text using the LLaMA model with a custom prompt.

        :param text: The text to classify.
        :param current_index: The index of the current segment in the DataFrame.
        :param df: DataFrame containing the transcription data.
        :param context_length: Number of previous and future lines to include in the prompt.
        :return: Predicted label for the text.
        """
        self.logger.debug(f"Classifying text at index {current_index}: {text[:50]}...")
        if not isinstance(text, str):
            self.logger.warning("Input is not a string. Returning 'No Voice Detected'.")
            return "No Voice Detected"
        if text.strip() == "":
            self.logger.info("Empty text. Returning 'No Voice Detected'.")
            return "No Voice Detected"

        if context_length is None:
            context_length = self.context_length

        prompt = self.build_prompt(current_index, df, context_length)

        try:
            classification = self.classify_line(prompt)
            self.logger.info(f"Predicted classification: {classification}")
            return classification
        except Exception as e:
            self.logger.error(f"Error during classification: {e}", exc_info=True)
            return "Error"

    def process_file(self, file_name):
        """
        Process a single transcription file to classify text segments and save the results as a log file.
        :param file_name: The name of the file to process.
        :return: DataFrame with classification results.
        """
        transcription_file = os.path.join(self.transcription_path, file_name.replace('.mp3', '_transcription.csv'))
        start_time = time.time()
        self.logger.info(f"Processing file: {file_name}")
        try:
            transcription_df = pd.read_csv(transcription_file)
            self.logger.info(f"Loaded transcription file: {transcription_file} with {len(transcription_df)} rows.")
            if 'Text' not in transcription_df.columns:
                raise ValueError(f"The transcription file {transcription_file} must contain a 'Text' column.")

            transcription_df["Text"] = transcription_df["Text"].fillna("").astype(str)
            transcription_df["file_name"] = file_name

            transcription_df["flow_label"] = transcription_df.apply(
                lambda row: self.classify_with_llama(
                    row["Text"], current_index=row.name, df=transcription_df, context_length=3
                ),
                axis=1
            )

            log_file_name = os.path.join(self.logs_folder_path, file_name.replace('.mp3', '_log.csv'))
            transcription_df.to_csv(log_file_name, index=False)
            self.logger.info(f"Processed file {file_name} in {time.time() - start_time:.2f} seconds. Log saved to {log_file_name}")
            return transcription_df
        except Exception as e:
            self.logger.error(f"Error processing file {file_name}: {e}", exc_info=True)
            return pd.DataFrame()

    def process_all_files(self):
        """
        Process all transcriptions in the metadata DataFrame, ensuring consistency by joining results.
        :return: Consolidated classification results for all files.
        """
        files_to_process = self.metadata_df[
            (self.metadata_df["transcription_available"] == True) &
            (self.metadata_df["log_available"] == False)
        ]["file_name"]
        self.logger.info(f"Starting processing for {len(files_to_process)} files.")
        try:
            torch.multiprocessing.set_start_method("spawn", force=True)
            with ProcessPoolExecutor(max_workers=self.cores) as executor:
                futures = {executor.submit(self.process_file, file_name): file_name for file_name in files_to_process}
                for future in futures:
                    try:
                        result = future.result()
                        if not result.empty:
                            merged_result = result.merge(
                                self.metadata_df, on="file_name", how="inner"
                            )
                            self.classification_data = pd.concat(
                                [self.classification_data, merged_result], ignore_index=True
                            )
                    except Exception as e:
                        self.logger.error(f"Error during parallel processing for file {futures[future]}: {e}", exc_info=True)
        except RuntimeError as e:
            self.logger.error(f"Multiprocessing failed with error: {e}. Falling back to sequential processing.")
            for file_name in files_to_process:
                result = self.process_file(file_name)
                if not result.empty:
                    merged_result = result.merge(
                        self.metadata_df, on="file_name", how="inner"
                    )
                    self.classification_data = pd.concat(
                        [self.classification_data, merged_result], ignore_index=True
                    )
        self.logger.info(f"Completed processing for all files.")

    def get_data(self):
        """
        Retrieve the consolidated classification data.
        :return: DataFrame with classification results.
        """
        self.logger.info("Retrieving consolidated classification data.")
        return self.classification_data
