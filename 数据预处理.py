import pandas as pd
import numpy as np
import ast

INPUT_EXCEL = "留言板爬虫20240101-20251209.xlsx"
OUTPUT_CSV = "留言板数据.csv"
OUTPUT_PKL = "processed_data.pkl"

def convert_excel_to_csv():
    df = pd.read_excel(INPUT_EXCEL)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"✅ 已保存 CSV：{OUTPUT_CSV}")

def parse_tags(tag_str):
    try:
        tags = ast.literal_eval(tag_str)
        if isinstance(tags, tuple):
            tags = [t.strip() for t in tags]
            while len(tags) < 3:
                tags.append(None)
            return tags[:3]
    except:
        pass
    return None, None, None

def preprocess_data():
    df = pd.read_csv(OUTPUT_CSV, encoding="utf-8-sig")

    column_mapping = {
        df.columns[0]: 'user_name',
        df.columns[1]: 'message_id',
        df.columns[2]: 'message_time',
        df.columns[3]: 'tags',
        df.columns[4]: 'location',
        df.columns[5]: 'title',
        df.columns[6]: 'content',
        df.columns[7]: 'reply_unit',
        df.columns[8]: 'reply_time',
        df.columns[9]: 'reply_content',
        df.columns[10]: 'source'
    }
    df = df.rename(columns=column_mapping)

    tag_data = df['tags'].apply(parse_tags)
    df['message_type'] = tag_data.apply(lambda x: x[0])
    df['theme'] = tag_data.apply(lambda x: x[1])
    df['status'] = tag_data.apply(lambda x: x[2])

    df['message_time'] = pd.to_datetime(df['message_time'], errors='coerce')
    df['reply_time'] = pd.to_datetime(df['reply_time'], errors='coerce')

    df['duration_days'] = (df['reply_time'] - df['message_time']).dt.total_seconds() / (24 * 3600)
    df.loc[df['duration_days'] < 0, 'duration_days'] = np.nan

    df['month'] = df['message_time'].dt.to_period('M')
    df['hour'] = df['message_time'].dt.hour
    df['day_of_week'] = df['message_time'].dt.day_name()
    df['reply_unit'] = df['reply_unit'].fillna("未回复")

    df.to_pickle(OUTPUT_PKL)
    print(f"✅ 已保存 pkl：{OUTPUT_PKL}")

if __name__ == "__main__":
    convert_excel_to_csv()
    preprocess_data()
