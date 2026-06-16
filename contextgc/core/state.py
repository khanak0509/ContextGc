import json
import os
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate


class MemorySchema(BaseModel):
    current_topic: str = Field(default="", description="One sentence on what the user and AI are currently discussing")
    core_entities: list[str] = Field(default_factory=list, description="Important people, places, concepts, or tools mentioned")
    key_facts: list[str] = Field(default_factory=list, description="Important facts, decisions, or details established")
    user_preferences: list[str] = Field(default_factory=list, description="How the user likes the AI to respond")


EXTRACT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are a memory consolidation engine. Keep all previous facts unless contradicted. Extract strictly factual key points."),
    ("user", "Read these conversation messages and update the memory state.\n\nCurrent state:\n{state}\n\nMessages being archived:\n{convo}\n\nExtract and return updated memory.")
])


class CoreState:
    def __init__(self, model, state_path=None):
        self.model = model
        self.state_path = state_path
        self.data = self.load()
        self.llm = ChatOllama(model=model)

    def load(self):
        if self.state_path and os.path.exists(self.state_path):
            try:
                with open(self.state_path) as f:
                    return MemorySchema.model_validate(json.load(f))
            except Exception:
                pass
        return MemorySchema()

    def save(self):
        if not self.state_path:
            return
        try:
            with open(self.state_path, "w") as f:
                json.dump(self.data.model_dump(), f, indent=2)
        except Exception:
            pass

    def extract_from(self, msgs):
        if not msgs:
            return

        convo = "\n".join(f"{m['role']}: {m['content'][:250]}" for m in msgs[:10])
        try:
            chain = EXTRACT_PROMPT | self.llm.with_structured_output(MemorySchema)
            result = chain.invoke({
                "state": json.dumps(self.data.model_dump(), indent=2),
                "convo": convo
            })

            # merge: keep existing facts, add new ones, deduplicate
            merged_facts = list(dict.fromkeys(self.data.key_facts + result.key_facts))
            merged_entities = list(dict.fromkeys(self.data.core_entities + result.core_entities))
            merged_prefs = list(dict.fromkeys(self.data.user_preferences + result.user_preferences))

            self.data = MemorySchema(
                current_topic=result.current_topic or self.data.current_topic,
                core_entities=merged_entities,
                key_facts=merged_facts,
                user_preferences=merged_prefs,
            )
            self.save()

        except Exception:
            pass  # keep current state on any failure — never crash

    def as_system_message(self):
        has_content = self.data.current_topic or self.data.key_facts
        if not has_content:
            return None

        lines = ["[Memory] Persistent context from this conversation:"]
        if self.data.current_topic:
            lines.append(f"Topic: {self.data.current_topic}")
        if self.data.core_entities:
            lines.append(f"Entities: {', '.join(self.data.core_entities)}")
        if self.data.key_facts:
            lines.append("Key facts:")
            for fact in self.data.key_facts:
                lines.append(f"  - {fact}")
        if self.data.user_preferences:
            lines.append(f"Preferences: {', '.join(self.data.user_preferences)}")
        return "\n".join(lines)

    def is_empty(self):
        return not (self.data.project_focus or self.data.key_facts)
