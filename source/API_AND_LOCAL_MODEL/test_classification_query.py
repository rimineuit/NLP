import psycopg2
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Khởi tạo mô hình embedding
model = SentenceTransformer("bkai-foundation-models/vietnamese-bi-encoder")
model.max_seq_length = 128

# Kết nối PostgreSQL
DB_NAME = "postgres"
DB_USER = "postgres"
DB_PASSWORD = "minh0985362932"
DB_HOST = "localhost"
DB_PORT = "5432"

conn = psycopg2.connect(
    dbname=DB_NAME,
    user=DB_USER,
    password=DB_PASSWORD,
    host=DB_HOST,
    port=DB_PORT
)
cursor = conn.cursor()
# Lấy tất cả embeddings từ PostgreSQL
cursor.execute("SELECT labels, embeddings FROM embeddings_data")
rows = cursor.fetchall()

# Chuyển dữ liệu thành dict {label: [embedding1, embedding2, ...]}
train_groups = {}

for label, embedding_json in rows:
    embedding = np.array(json.loads(embedding_json))  # Chuyển JSON thành numpy array
    if label not in train_groups:
        train_groups[label] = []
    train_groups[label].append(embedding)

# Chuyển list thành numpy array để tính toán nhanh hơn
for label in train_groups:
    train_groups[label] = np.array(train_groups[label])
def predict_label(test_emb):
    similarities = {}
    for label, train_embeds in train_groups.items():
        if len(train_embeds) > 0:
            cos_sim = cosine_similarity([test_emb], train_embeds).mean()
            similarities[label] = cos_sim  
    return max(similarities, key=similarities.get)  # Chọn label có cosine cao nhất
def predict_from_text(text):
    # Chuyển văn bản thành embedding
    test_emb = model.encode(text)
    # Dự đoán nhãn
    predicted_label = predict_label(test_emb)
    return predicted_label

# Ví dụ
query = "bị đau bụng thì nên hạn chế ăn gì"
predicted_label = predict_from_text(query)
print(f"Label dự đoán: {predicted_label}")
