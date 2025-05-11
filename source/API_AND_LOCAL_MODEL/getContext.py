from pyvi import ViTokenizer
from dotenv import load_dotenv
import os
import psycopg2
from rank_bm25 import BM25Okapi
import json
load_dotenv()

class GetContextFromPostGreSQL:
    def __init__(self):
        DB_NAME = os.getenv("DB_NAME")
        DB_USER = os.getenv("DB_USER")
        DB_PASSWORD = os.getenv("DB_PASSWORD")
        DB_HOST = os.getenv("DB_HOST")
        DB_PORT = os.getenv("DB_PORT")
        self.conn = psycopg2.connect(
                dbname=DB_NAME,
                user=DB_USER,
                password=DB_PASSWORD,
                host=DB_HOST,
                port=DB_PORT
            )
        

    def init_bm25(self):
        cursor = self.conn.cursor()
        cursor.execute("SELECT id, tf_lower FROM word_frequency_health_paragraphs")
        rows = cursor.fetchall()
        tokenized_corpus = []
        id_list = []

        for id_baiviet, tf_json in rows:
            if isinstance(tf_json, dict):  # Nếu đã là dict, không cần json.loads
                tf = tf_json
            elif isinstance(tf_json, str):  # Nếu là chuỗi, parse JSON
                tf = json.loads(tf_json)
            elif tf_json is None:  # Nếu là NULL trong PostgreSQL
                tf = {}
            else:
                raise TypeError(f"Dữ liệu không hợp lệ cho ID {id_baiviet}: {type(tf_json)}")
            
            tokenized_corpus.append(list(tf.keys()))
            id_list.append(id_baiviet)
            
        # Tạo BM25 với dữ liệu đã được xử lý
        bm25 = BM25Okapi(tokenized_corpus)  # Tạo BM25 với dữ liệu mới
        cursor.close()

        return bm25
    def preprocess_query(self, query):
        # Tokenize the query using ViTokenizer
        tokenized_query = ViTokenizer.tokenize(query.lower()).split()
        return tokenized_query
    
    
    def get_context(self, context_id):
        cursor = self.conn.cursor()
        cursor.execute("SELECT context FROM contexts WHERE id = %s", (context_id,))
        context = cursor.fetchone()
        if context:
            return context[0]
        else:
            return None
        
    def rerank_chunk(self):
        """""""