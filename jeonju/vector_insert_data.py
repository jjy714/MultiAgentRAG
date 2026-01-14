import os
import util.config
from pymilvus import connections, Collection
import pandas as pd
from openai import OpenAI

milvus_url = util.config.DATABASE_URL
collection_name = util.config.COLLECTION
database = util.config.DATABASE

#pymilvus.connections.connect(database, uri="http://27.96.148.214:19530")
connections.connect(db_name=database, host="27.96.148.214", port="19530")
collection = Collection(name=collection_name)
#print(collection.schema)

EMBED_MODEL = "text-embedding-3-small"

df = pd.read_excel('전주시 표 누락건.xlsx', sheet_name='Sheet2')

headers = df.columns.tolist()
#print('헤더')
#print(headers)
#print(headers[6])

for index, row in df.iterrows():
    # 특정 열에 접근할 수 있습니다. (열 이름이 숫자나 공백 없이 깨끗한 경우에만 가능)
    #print(f"Index: {index}, Data: {row}")
    vector_list = []
    content_list = []
    data = []
    _content = f"**제목**: {row[headers[1]]}\n\n**파일명**: {row[headers[0]]}\n\n**내용**: \n{row[headers[2]]}"

    _embedding = (
        OpenAI()
        .embeddings.create(input=_content, model=EMBED_MODEL)
        .data[0]
        .embedding
    )

    vector_list.append(_embedding)
    content_list.append(_content)
    data = [content_list, vector_list]

    #print(data)
    #print(_embedding)
    collection.insert(data)