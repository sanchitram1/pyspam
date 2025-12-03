# ------------------------------------------------------
# Step 6: Time offline features
#   - t_time_of_day (morning, evening, afternoon, night)
# ------------------------------------------------------
import math
import numpy as np
import pandas as pd


def bucket_to_time_of_day(bucket):
    if 'evening' in bucket:
        return 'evening'
    elif 'morning' in bucket:
        return 'morning'
    elif 'night' in bucket:
        return 'night'
    elif "afternoon" in bucket:
        return 'afternoon'
    else:
        return 'unknown'
    
def handle_time(df:pd.DataFrame):
    df['t_time_of_day'] = df["t_time_of_day_bucket"].apply(bucket_to_time_of_day)
    return df    

