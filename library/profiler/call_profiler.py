from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import pandas as pd
from pydub.utils import mediainfo
from pydub import AudioSegment


class CallProfiler:
    def __init__(self):
        self.call_data = pd.DataFrame(columns=["file_name", "format", "length_seconds", "transcription_available", "log_available"])
        self.last_run_params = None  # Store parameters for the last run

    def _log(self, message, level, log_level):
        """
        Log a message based on the current log level.
        """
        levels = ["off", "low", "medium", "high"]
        if levels.index(level) <= levels.index(log_level):
            print(message)

    def _process_file(self, file_path, log_level):
        """
        Processes a single file: extracts metadata and calculates length.
        """
        file_name = os.path.basename(file_path)
        try:
            file_info = mediainfo(file_path)
            file_format = file_info.get('format_name', 'unknown')
            audio = AudioSegment.from_file(file_path)
            length_seconds = len(audio) / 1000  # Convert milliseconds to seconds
            self._log(f"Processed {file_name} (format: {file_format}, length: {length_seconds:.2f}s)", "low", log_level)
            return {"file_name": file_name, "format": file_format, "length_seconds": length_seconds}
        except Exception as e:
            self._log(f"Error processing {file_name}: {e}", "high", log_level)
            return None

    def profile_calls(self, calls_folder_path, transcripts_folder_path, logs_folder_path, original_metadata_file, generate_csv=False, log_level="low", cores=4, persistent_mode=False):
        """
        Profiles audio calls and includes checks for transcriptions, logs, and metadata.

        Parameters:
            calls_folder_path (str): Path to the folder containing audio files.
            transcripts_folder_path (str): Path to the folder containing transcription files.
            logs_folder_path (str): Path to the folder containing log files.
            original_metadata_file (str): Path to the metadata CSV file.
            generate_csv (bool): Whether to generate a CSV of the results (default is False).
            log_level (str): Logging level for the function ('off', 'low', 'medium', 'high'). Default is 'low'.
            cores (int): Number of cores to use for parallel processing. Default is 4.
            persistent_mode (bool): If True, uses persistent mode for loading/updating a stored metadata CSV file.
        """
        self.last_run_params = {
            "calls_folder_path": calls_folder_path,
            "transcripts_folder_path": transcripts_folder_path,
            "logs_folder_path": logs_folder_path,
            "original_metadata_file": original_metadata_file,
            "generate_csv": generate_csv,
            "log_level": log_level,
            "cores": cores,
            "persistent_mode": persistent_mode,
        }

        if log_level not in ["off", "low", "medium", "high"]:
            raise ValueError("Invalid log_level. Choose from 'off', 'low', 'medium', 'high'.")

        self._log(f"Starting profiling in folder: {calls_folder_path} using {cores} core(s). Persistent mode: {persistent_mode}", "low", log_level)

        # Persistent CSV setup
        assets_folder = os.path.join(os.getcwd(), "assets", "call_metadata")
        os.makedirs(assets_folder, exist_ok=True)
        persistent_csv_path = os.path.join(assets_folder, "persistent_call_data.csv")
        generated_csv_name = "profiler_generated_metadata.csv"

        # Load metadata CSV
        try:
            metadata_df = pd.read_csv(original_metadata_file, usecols=["file_name", "state", "description"])
        except FileNotFoundError:
            self._log(f"Metadata CSV not found at {original_metadata_file}. Aborting operation.", "high", log_level)
            return

        # Rename state column to category for mapping
        metadata_df.rename(columns={"state": "category"}, inplace=True)

        # Load or reset metadata
        if persistent_mode and os.path.exists(persistent_csv_path):
            self._log(f"Loading existing metadata from {persistent_csv_path}", "low", log_level)
            try:
                self.call_data = pd.read_csv(persistent_csv_path)
            except Exception as e:
                self._log(f"Error loading persistent CSV: {e}", "high", log_level)
                return
        else:
            self._log(f"Resetting metadata: Persistent mode is {persistent_mode}", "medium", log_level)
            self.call_data = pd.DataFrame(columns=["file_name", "format", "length_seconds", "transcription_available", "log_available"])

        # Pre-filter files
        processed_files = set(self.call_data["file_name"])
        file_paths = [
            os.path.join(calls_folder_path, f) for f in os.listdir(calls_folder_path)
            if os.path.isfile(os.path.join(calls_folder_path, f)) and f not in processed_files
        ]

        self._log(f"Found {len(file_paths)} new files to process.", "medium", log_level)

        # Process files in parallel
        results = []
        with ProcessPoolExecutor(max_workers=cores) as executor:
            futures = {executor.submit(self._process_file, fp, log_level): fp for fp in file_paths}
            for future in as_completed(futures):
                result = future.result()
                if result:
                    results.append(result)

        # Update DataFrame
        if results:
            self.call_data = pd.concat([self.call_data, pd.DataFrame(results)], ignore_index=True)

        # Drop conflicting columns before merging
        self.call_data = self.call_data.drop(columns=["category", "description"], errors="ignore")

        # Perform a left join to include category and description from metadata
        self.call_data = pd.merge(
            self.call_data,
            metadata_df[["file_name", "category", "description"]],
            on="file_name",
            how="left"
        )

        # Update transcription and log availability for all files
        self.call_data["transcription_available"] = self.call_data["file_name"].apply(
            lambda x: os.path.exists(os.path.join(transcripts_folder_path, f"{os.path.splitext(x)[0]}_transcription.csv"))
        )
        self.call_data["log_available"] = self.call_data["file_name"].apply(
            lambda x: os.path.exists(os.path.join(logs_folder_path, f"{os.path.splitext(x)[0]}_log.csv"))
        )

        # Save the persistent CSV (always overwrite if not in persistent mode)
        try:
            self.call_data.to_csv(persistent_csv_path, index=False)
            self._log(f"Persistent CSV saved at {persistent_csv_path}", "low", log_level)
        except Exception as e:
            self._log(f"Error saving persistent CSV: {e}", "high", log_level)

        # Save to fixed CSV in the current working directory if generate_csv is True
        if generate_csv:
            try:
                output_csv_path = os.path.join(os.getcwd(), generated_csv_name)
                self.call_data.to_csv(output_csv_path, index=False)
                self._log(f"Generated CSV saved at {output_csv_path}", "low", log_level)
            except Exception as e:
                self._log(f"Error saving generated CSV: {e}", "high", log_level)

        self._log("Profiling complete. DataFrame updated.", "low", log_level)

    def update(self, call_metadata_df=None):
        """
        Update the profiler metadata.

        Parameters:
            call_metadata_df (DataFrame or None): If provided, updates the current metadata in place. If None, reruns the last parameters.
        Returns:
            DataFrame: The updated rows from the provided DataFrame.
        """
        if call_metadata_df is not None:
            if not isinstance(call_metadata_df, pd.DataFrame):
                raise ValueError("call_metadata_df must be a pandas DataFrame.")
            
            # Update availability flags for the provided DataFrame
            call_metadata_df["transcription_available"] = call_metadata_df["file_name"].apply(
                lambda x: os.path.exists(os.path.join(self.last_run_params["transcripts_folder_path"], f"{os.path.splitext(x)[0]}_transcription.csv"))
            )
            call_metadata_df["log_available"] = call_metadata_df["file_name"].apply(
                lambda x: os.path.exists(os.path.join(self.last_run_params["logs_folder_path"], f"{os.path.splitext(x)[0]}_log.csv"))
            )
            
            # Update the main call_data DataFrame
            self.call_data.update(call_metadata_df)

            # Ensure the returned dataset matches the updated dataset passed in
            call_metadata_df = self.call_data[self.call_data["file_name"].isin(call_metadata_df["file_name"])]
            self._log("Call data updated in place from provided DataFrame.", "medium", "low")
            
            return call_metadata_df

        elif self.last_run_params:
            self.profile_calls(**self.last_run_params)
        else:
            raise ValueError("No previous run parameters found. Run profile_calls() first.")

    def get_data(self):
        """
        Returns the current call data as a DataFrame.
        """
        return self.call_data
