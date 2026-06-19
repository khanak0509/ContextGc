import os
import sys
import json
import time
import uuid
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from contextgc.core.archive import MessageArchive
from contextgc.core.eviction import EvictionOrchestrator
from contextgc.core.scorer import MessageScorer
from contextgc.core.state import CoreState, MemorySchema


def make_msg(role, content, step=0):
    return {
        "id": str(uuid.uuid4()),
        "role": role,
        "content": content,
        "timestamp": time.time() + step,
        "metadata": {},
    }


def redis_convo(n):
    msgs = [make_msg("system", "You are a coding assistant focused on Redis and PostgreSQL.", step=-1)]
    for i in range(n):
        msgs.append(make_msg("user",
            f"Step {i}: Debug Redis cache miss. PostgreSQL fallback needed. JWT error AUTH_{i}. "
            f"Session TTL expired at 300s causing 401 on route /api/session/{i}.",
            step=i))
        msgs.append(make_msg("assistant",
            f"Step {i}: Inspected redis.get() call. No fallback to PG. "
            f"AUTH_SESSION_MISS_{i} should map to 503 not 401. Adding exponential backoff.",
            step=i))
    return msgs


class TestContextGC(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp.close()
        self.db = self.tmp.name
        self.state_file = self.db.replace(".db", "_state.json")

    def tearDown(self):
        for path in [self.db, self.state_file]:
            if os.path.exists(path):
                os.remove(path)

    def gc(self, max_tokens=800):
        return EvictionOrchestrator(
            max_tokens=max_tokens,
            state_path=self.state_file,
        )

    # ── Test 1: token count drops after GC runs ───────────────────────────────
    def test_token_reduction(self):
        msgs = redis_convo(25)
        o = self.gc(max_tokens=600)
        before = o.count_tokens(msgs)
        out = o.process(list(msgs))
        after = o.count_tokens(out)
        self.assertLess(after, before, "GC should reduce total tokens")
        print(f"PASS token_reduction: {before} -> {after} tokens "
              f"({(before - after) / before * 100:.1f}% reduction)")

    # ── Test 2: original system prompt always survives eviction ───────────────
    def test_system_prompt_preserved(self):
        msgs = redis_convo(30)
        o = self.gc(max_tokens=500)
        out = o.process(list(msgs))
        sys_contents = " ".join(m["content"] for m in out if m["role"] == "system")
        self.assertIn("coding assistant", sys_contents.lower())
        print("PASS system_prompt_preserved: original system prompt found after eviction")

    # ── Test 3: archived message recalled by semantic similarity ──────────────
    def test_recall(self):
        from langchain_core.vectorstores import InMemoryVectorStore
        from contextgc.core.scorer import MessageScorer
        scorer = MessageScorer()
        archive = MessageArchive(InMemoryVectorStore(embedding=scorer.model))

        goal = "Redis AUTH_SESSION_MISS PostgreSQL fallback"
        goal_vec = scorer.compute_embeddings([goal])[0]

        msg = make_msg("user",
            "CRITICAL: Redis AUTH_SESSION_MISS error — session expired, PG fallback required immediately.")
        vec = scorer.compute_embeddings([msg["content"]])[0]
        archive.archive_message(msg["id"], msg["role"], msg["content"], vec, msg["timestamp"], {})

        hits = archive.recall_relevant_messages(goal_vec, threshold=0.5)
        self.assertTrue(any("AUTH_SESSION_MISS" in item["content"] for item, _ in hits))
        print("PASS recall: archived Redis message retrieved by semantic similarity")

    # ── Test 4: block eviction sends oldest messages to archive ───────────────
    def test_block_eviction(self):
        msgs = redis_convo(15)
        o = self.gc(max_tokens=500)
        first_user_id = msgs[1]["id"]   # first user message (after system)

        out = o.process(list(msgs))

        # eviction counter must have fired
        self.assertGreater(o.total_evictions, 0, "Expected at least one eviction")

        # the first user message should now be in the archive, not in context
        out_ids = {m["id"] for m in out}
        self.assertNotIn(first_user_id, out_ids, "Oldest message should be archived, not in context")

        # archive must have content
        import sqlite3
        with sqlite3.connect(self.db) as conn:
            count = conn.execute("SELECT COUNT(*) FROM archived_messages").fetchone()[0]
        self.assertGreater(count, 0, "Archive should have messages after eviction")

        print(f"PASS block_eviction: {o.total_evictions} messages moved to archive, {count} rows in DB")

    # ── Test 5: a breadcrumb stays in context after eviction ──────────────────
    def test_breadcrumb_left_in_context(self):
        msgs = redis_convo(20)
        o = self.gc(max_tokens=500)
        out = o.process(list(msgs))

        breadcrumbs = [m for m in out if m.get("metadata", {}).get("_gc_breadcrumb")]
        self.assertGreater(len(breadcrumbs), 0, "Expected a breadcrumb system message")
        self.assertIn("Archived", breadcrumbs[0]["content"])
        print(f"PASS breadcrumb: '{breadcrumbs[0]['content'][:60]}...'")

    # ── Test 6: CoreState injection — injected state visible in context ────────
    def test_core_state_injected(self):
        msgs = redis_convo(20)
        o = self.gc(max_tokens=500)

        o.core_state.data = MemorySchema(
            current_topic="Debugging Redis session cache AUTH_SESSION_MISS error",
            core_entities=["Redis", "PostgreSQL", "JWT"],
            key_facts=["AUTH_SESSION_MISS maps to 503", "Session TTL is 300s"],
            user_preferences=[],
        )
        out = o.process(list(msgs))

        sys_texts = " ".join(m["content"] for m in out if m["role"] == "system")
        self.assertIn("[Memory]", sys_texts)
        self.assertIn("Redis", sys_texts)
        self.assertIn("AUTH_SESSION_MISS", sys_texts)
        print("PASS core_state_injected: [Memory] block visible in context after eviction")

    # ── Test 7: state persists to disk and reloads correctly ──────────────────
    def test_state_persistence(self):
        o = self.gc()
        o.core_state.data = MemorySchema(
            current_topic="Fix auth in production API",
            core_entities=["Redis", "FastAPI"],
            key_facts=["401 caused by Redis TTL expiry", "JWT uses HS256"],
            user_preferences=["prefers Python"],
        )
        o.core_state.save()

        # create a brand new orchestrator pointing at the same state file
        o2 = EvictionOrchestrator(archive_path=self.db, state_path=self.state_file)
        self.assertEqual(o2.core_state.data.current_topic, "Fix auth in production API")
        self.assertIn("Redis", o2.core_state.data.core_entities)
        self.assertIn("JWT uses HS256", o2.core_state.data.key_facts)
        print("PASS state_persistence: CoreState saved and reloaded correctly across sessions")

    # ── Test 8: no Ollama call when under the token limit ─────────────────────
    def test_no_extraction_under_limit(self):
        # 3 short messages will be way under any limit
        msgs = [
            make_msg("system", "You are a helpful assistant."),
            make_msg("user", "hi"),
            make_msg("assistant", "hello"),
        ]
        o = self.gc(max_tokens=10000)
        before_topic = o.core_state.data.current_topic
        out = o.process(list(msgs))

        self.assertEqual(o.total_evictions, 0)
        self.assertEqual(o.core_state.data.current_topic, before_topic)
        print("PASS no_extraction_under_limit: Ollama not called when tokens are under limit")


if __name__ == "__main__":
    print("Running ContextGC intensive tests...\n")
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestContextGC)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    print()
    total = result.testsRun
    failed = len(result.failures) + len(result.errors)
    passed = total - failed
    print(f"{passed}/{total} tests passed.")
    if failed:
        sys.exit(1)
