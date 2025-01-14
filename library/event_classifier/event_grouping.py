import pandas as pd

def group_consecutive_events(df, start_col, end_col, text_col, speaker_col, event_col, case_id_col, drop_cols=[]):
    """
    Groups consecutive events by case_id and event, concatenates text with speaker, 
    and combines timestamps.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        start_col (str): Column name for the start timestamp.
        end_col (str): Column name for the end timestamp.
        text_col (str): Column name for the text.
        speaker_col (str): Column name for the speaker.
        event_col (str): Column name for the event.
        case_id_col (str): Column name for the case ID.
        drop_cols (list): List of column names to drop from the final DataFrame.
    
    Returns:
        pd.DataFrame: Processed DataFrame with grouped and concatenated events.
    """
    # Ensure timestamps are in datetime format
    df[start_col] = pd.to_datetime(df[start_col])
    df[end_col] = pd.to_datetime(df[end_col])
    
    # Add a helper column to identify consecutive groups of the same event
    df["group"] = (df[event_col] != df[event_col].shift()) | (df[case_id_col] != df[case_id_col].shift())
    df["group"] = df["group"].cumsum()

    # Group by case_id, event, and the helper group column
    grouped_df = (
        df.groupby([case_id_col, "group", event_col])
        .agg(
            **{
                start_col: (start_col, "min"),  # Earliest start
                end_col: (end_col, "max"),      # Latest end
                text_col: (text_col, lambda x: " ".join(
                    f"{df.loc[i, speaker_col]}: {text}" for i, text in zip(x.index, x)
                ))
            }
        )
        .reset_index()
    )

    # Drop unnecessary columns
    grouped_df = grouped_df.drop(columns=drop_cols, errors="ignore")

    return grouped_df
