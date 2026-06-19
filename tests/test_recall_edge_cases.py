import os
import sys
import time
import uuid
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from contextgc.core.archive import MessageArchive
from contextgc.core.eviction import EvictionOrchestrator
from contextgc.core.scorer import MessageScorer

def make_msg(role, content):
    return {
        "id": str(uuid.uuid4()),
        "role": role,
        "content": content,
        "timestamp": time.time(),
        "metadata": {},
    }

class TestRecallEdgeCases(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp.close()
        self.db = self.tmp.name
        self.state_file = self.db.replace(".db", "_state.json")
        self.archive = MessageArchive(self.db)
        self.scorer = MessageScorer()

    def tearDown(self):
        if os.path.exists(self.db):
            os.remove(self.db)
        if os.path.exists(self.state_file):
            os.remove(self.state_file)

    def get_gc(self, max_tokens=1000, watermark=0.8):
        return EvictionOrchestrator(
            model="llama3.1",
            max_tokens=max_tokens,
            watermark=watermark,
            state_path=self.state_file
        )

    def _archive_messages(self, messages):
        for msg in messages:
            vec = self.scorer.compute_embeddings([msg["content"]])[0]
            self.archive.archive_message(
                msg["id"], msg["role"], msg["content"], vec, msg["timestamp"], {}
            )

    # 1. Semantic Synonym Recall (No exact keyword match)
    def test_semantic_synonym_recall(self):
        # Database has "TTL is 300s"
        self._archive_messages([
            make_msg("assistant", "I have updated the Redis configuration. The session TTL is now set to 300s to prevent early logouts.")
        ])
        
        # User asks about "timeout" and "expiration" (synonyms, no exact match)
        query = "What did we set the session timeout or expiration to?"
        vec = self.scorer.compute_embeddings([query])[0]
        
        hits = self.archive.recall_relevant_messages(vec, threshold=0.45)
        self.assertTrue(len(hits) > 0, "Failed to recall message using synonyms")
        self.assertIn("300s", hits[0][0]["content"])
        print("PASS semantic_synonyms: Found 'TTL is 300s' using query 'timeout or expiration'")

    # 2. Noise Resistance (Needle in a Haystack)
    def test_needle_in_haystack(self):
        # Generate 100 irrelevant messages about frontend CSS
        noise = [make_msg("user", f"Change the button color to {i} blue and add padding.") for i in range(100)]
        
        # The 1 critical message about the database password
        needle = make_msg("assistant", "The production database password is 'hunter2'. Please don't share this.")
        
        self._archive_messages(noise + [needle])
        
        # Search query
        query = "What is the secret key for the DB?"
        vec = self.scorer.compute_embeddings([query])[0]
        
        hits = self.archive.recall_relevant_messages(vec, threshold=0.50)
        
        self.assertTrue(len(hits) > 0, "Failed to find needle in haystack")
        self.assertIn("hunter2", hits[0][0]["content"], "Retrieved wrong message")
        print(f"PASS needle_in_haystack: Successfully found 1 DB password among 100 CSS messages")

    # 3. Cross-Topic Disambiguation
    def test_cross_topic_disambiguation(self):
        # Two very similar messages, but different technologies
        msg_redis = make_msg("assistant", "The Redis authentication error is caused by a missing API token in the header.")
        msg_postgres = make_msg("assistant", "The PostgreSQL authentication error is caused by a wrong password in the .env file.")
        
        self._archive_messages([msg_redis, msg_postgres])
        
        # Ask specifically about Postgres
        query = "Why is the postgres database failing to authenticate?"
        vec = self.scorer.compute_embeddings([query])[0]
        
        hits = self.archive.recall_relevant_messages(vec, threshold=0.50)
        
        self.assertTrue(len(hits) > 0)
        # It should rank the postgres message highest
        self.assertIn("PostgreSQL", hits[0][0]["content"])
        print("PASS disambiguation: Correctly distinguished PostgreSQL auth error from Redis auth error")

    # 4. End-to-End Recall via Orchestrator
    def test_orchestrator_recall_injection(self):
        # Put an old message in the archive
        self._archive_messages([
            make_msg("assistant", "We decided to use FastAPI instead of Flask for performance reasons.")
        ])
        
        # Create orchestrator
        o = EvictionOrchestrator(archive_path=self.db, state_path=self.db.replace(".db", "_state.json"))
        
        # Simulate active conversation where user asks about the framework decision
        active_msgs = [
            make_msg("system", "You are an AI."),
            make_msg("user", "Why did we choose FastAPI again?")
        ]
        
        out_msgs = o.process(list(active_msgs))
        
        # The recalled message should be injected AFTER the system prompt and BEFORE the user message
        recalled_content = " ".join(m["content"] for m in out_msgs)
        self.assertIn("We decided to use FastAPI instead of Flask", recalled_content)
        
        # Archive should no longer contain the message (it was moved back to active context)
        with sqlite3_connect(self.db) as conn:
            count = conn.execute("SELECT COUNT(*) FROM archived_messages").fetchone()[0]
        self.assertEqual(count, 0, "Message was not deleted from archive after recall")
        print("PASS orchestrator_recall: Message seamlessly injected back into active context and removed from DB")

def sqlite3_connect(db_path):
    import sqlite3
    return sqlite3.connect(db_path)

if __name__ == "__main__":
    print("Running Semantic Recall Edge Case Tests...\n")
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestRecallEdgeCases)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    print()
    if result.wasSuccessful():
        sys.exit(0)
    else:
        sys.exit(1)
