import os
import json
import datetime
import requests
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
import requests
import logging
from io import StringIO

url = 'https://api.producthunt.com/v2/api/graphql'
headers = {
    'Authorization': 'Bearer 80lsUlx0NhU4tskes86Tj7oF_YnQHv40YJf2ZOqWaa0',
    'Content-Type': 'application/json'
}

query = ("""
query fetchPosts($postedBefore: DateTime, $postedAfter: DateTime, $after: String){
  posts(postedBefore:$postedBefore, postedAfter:$postedAfter, after:$after) {
    pageInfo {
      hasNextPage,
      endCursor
    }
    nodes {
      id,
      name,
      votesCount,
      commentsCount,
      tagline,
      createdAt,
      description,
      topics{
        totalCount,
        nodes{
          name,
        }
      },
    }
  }
}
""")


def from_iso(iso_str):
    return datetime.strptime(iso_str, "%Y-%m-%d").date()

def crawl_one_day(date, after_cursor=""):
    posted_after = date.isoformat()
    posted_before = (date + timedelta(days=1)).isoformat()

    # print(posted_before, posted_after, after_cursor)
    variables = {"postedBefore": posted_before, "postedAfter": posted_after, "after": after_cursor}

    # result = client.execute(query, variable_values=variables)
    result = requests.post(url, json={'query': query, 'variables': variables}, headers=headers).json()

    if(result["data"]["posts"]["pageInfo"]["hasNextPage"]):
        next_nodes = crawl_one_day(date, result["data"]["posts"]["pageInfo"]["endCursor"])
        return result["data"]["posts"]["nodes"] + next_nodes

    return result["data"]["posts"]["nodes"] 

def extract_data():
    day_one = crawl_one_day(from_iso((datetime.now() - timedelta(days=4)).strftime("%Y-%m-%d")))

    return json.dumps(day_one)

# os.getenv("PH_ACCESS_TOKEN")
# Define default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'start_date': datetime(2024, 7, 19),
}

# Create the DAG object
dag = DAG(
    'data_pipeline',
    default_args=default_args,
    description='ETL pipeline for Product Hunt launch data',
    schedule='@daily',
    catchup=False,
)

def save_to_s3(data, s3_bucket, s3_key, aws_conn_id):

    import pandas as pd

    s3_hook = S3Hook(aws_conn_id)

    file_exists = s3_hook.check_for_key(s3_key, s3_bucket)

    if file_exists:
        existing_file_obj = s3_hook.get_key(s3_key, s3_bucket)
        string_data = existing_file_obj.get()['Body'].read().decode('utf-8')
        existing_df = pd.read_csv(StringIO(string_data))
    else:
        existing_df = pd.DataFrame()

    new_df = pd.json_normalize(json.loads(data))

    new_df.drop_duplicates(subset='id', keep="first", inplace=True)

    merged_df = new_df if existing_df.empty else pd.concat([new_df, existing_df])

    # Save the merged DataFrame to a temporary file
    merged_file_path = '/tmp/merged_products.csv'
    merged_df.to_csv(merged_file_path, index=False)


    # Upload the merged file back to S3
    s3_hook.load_file(filename=merged_file_path, key=s3_key,
                        bucket_name=s3_bucket, replace=True)


# Define the extract_data task
extract_task = PythonOperator(
    task_id='extract_data',
    python_callable=extract_data,
    dag=dag,
)

# Define the save_to_s3 task
save_task = PythonOperator(
    task_id='save_to_s3',
    python_callable=save_to_s3,
    op_args=['{{ ti.xcom_pull(task_ids="extract_data") }}'],
    op_kwargs={
        'execution_date': '{{ ts }}',
        's3_bucket': 'mlops-hot-or-meh2',
        's3_key': 'ProductHuntProducts.csv',
        'aws_conn_id': 's3_bucket_east',
        },
    dag=dag,
)

# Set the task dependencies
extract_task >> save_task