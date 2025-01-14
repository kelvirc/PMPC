import pandas as pd
import os
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.util import constants

class PMProcessor:
    def __init__(self, metadata_df, logs_folder_path):
        """
        Initialize the PMProcessor class.

        :param metadata_df: DataFrame with metadata about the files.
        :param logs_folder_path: Path to the directory containing log files.
        """
        self.metadata_df = metadata_df
        self.logs_folder_path = logs_folder_path

    def process_logs(self):
        """
        Process all log files specified in the metadata DataFrame.

        :return: DataFrame containing combined process mining data.
        """
        # Filter metadata_df for logs that are available
        files_to_process = self.metadata_df[self.metadata_df["log_available"] == True]["file_name"]

        all_logs = []

        for file_name in files_to_process:
            log_file_path = os.path.join(self.logs_folder_path, file_name.replace('.mp3', '_log.csv'))

            try:
                # Import log file using pandas
                df_log = pd.read_csv(log_file_path)

                # Ensure the necessary columns are present
                if not {'Start Time (s)', 'End Time (s)', 'Text', 'Speaker', 'flow_label'}.issubset(df_log.columns):
                    raise ValueError(f"Log file {log_file_path} is missing required columns.")

                # Rename columns to match process mining standards
                df_log = df_log.rename(columns={
                    'Start Time (s)': 'start_timestamp',
                    'End Time (s)': 'end_timestamp',
                    'flow_label': 'event',
                    'case:concept:name': 'case_id'
                })

                # Convert timestamps to datetime, starting from a default date
                base_date = pd.Timestamp("2024-01-01")
                df_log['start_timestamp'] = base_date + pd.to_timedelta(df_log['start_timestamp'], unit='s')
                df_log['end_timestamp'] = base_date + pd.to_timedelta(df_log['end_timestamp'], unit='s')

                # Use the file name as the case ID
                df_log['case_id'] = file_name.replace('_log.csv', '')

                # Append to the combined logs list
                all_logs.append(df_log)

            except Exception as e:
                print(f"Error processing log file {log_file_path}: {e}")

        # Combine all logs into a single DataFrame
        if all_logs:
            combined_logs = pd.concat(all_logs, ignore_index=True)
        else:
            combined_logs = pd.DataFrame()

        return combined_logs
