import json
import sqlite3
import numpy as np


class MessageArchive:
    def __init__(self, db_path):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS archived_messages (
                    id TEXT PRIMARY KEY,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    embedding TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    metadata TEXT
                )
            """)
            conn.commit()

    def archive_message(self, msg_id, role, content, embedding, timestamp, metadata=None):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO archived_messages "
                "(id, role, content, embedding, timestamp, metadata) VALUES (?,?,?,?,?,?)",
                (msg_id, role, content, json.dumps(embedding), timestamp, json.dumps(metadata or {}))
            )
            conn.commit()

    def recall_relevant_messages(self, goal_embedding, threshold=0.5):
        target = np.array(goal_embedding, dtype=np.float32)
        tnorm = np.linalg.norm(target)
        if tnorm == 0:
            return []

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT id, role, content, embedding, timestamp, metadata FROM archived_messages"
            ).fetchall()

        results = []
        for row in rows:
            vec = np.array(json.loads(row["embedding"]), dtype=np.float32)
            vnorm = np.linalg.norm(vec)
            if vnorm == 0:
                continue
            if len(vec) != len(target):
                continue
            sim = float(np.dot(target, vec) / (tnorm * vnorm))
            if sim >= threshold:
                results.append(({
                    "id": row["id"],
                    "role": row["role"],
                    "content": row["content"],
                    "timestamp": row["timestamp"],
                    "metadata": json.loads(row["metadata"]),
                }, sim))

        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def delete_messages(self, ids):
        if not ids:
            return
        with sqlite3.connect(self.db_path) as conn:
            placeholders = ",".join("?" for _ in ids)
            conn.execute(f"DELETE FROM archived_messages WHERE id IN ({placeholders})", tuple(ids))
            conn.commit()
