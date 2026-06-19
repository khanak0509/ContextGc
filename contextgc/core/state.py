import json
import logging
import os
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

logger = logging.getLogger(__name__)

class MemorySchema(BaseModel):
    current_topic: str = Field(default="", description="One sentence summary of what the user is currently discussing")
    core_entities: list[str] = Field(default_factory=list, description="Important people and places mentioned (names only, no code)")
    key_facts: list[str] = Field(default_factory=list, description="Permanent personal facts about the user: name, age, location, education, relationships, goals, career. NO code snippets.")
    user_preferences: list[str] = Field(default_factory=list, description="Only communication style preferences like 'prefers concise answers' or 'likes bullet points'. NOT code. NOT tasks.")

EXTRACT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are a memory consolidation engine. Your ONLY job is to extract ALL personal facts about the user. Always fill current_topic. Always extract names, locations, ages, relationships, preferences, and goals. Keep all previous facts unless contradicted."),
    ("user", "Read these conversation messages and update the memory state.\n\nCurrent state:\n{state}\n\nMessages being archived:\n{convo}\n\nExtract and return updated memory. current_topic MUST be filled.")
])

class CoreState:
    def __init__(self, model, state_path=None):
        self.model = model
        self.state_path = state_path
        self.data = self.load()
        self.llm = ChatOllama(model=model)
        self.chain = EXTRACT_PROMPT | self.llm.with_structured_output(MemorySchema)

    def load(self):
        if self.state_path:
            if os.path.exists(self.state_path):
                try:
                    with open(self.state_path) as f:
                        data = json.load(f)
                        return MemorySchema.model_validate(data)
                except Exception:
                    pass
                    
        return MemorySchema()

    def save(self):
        if not self.state_path:
            return
            
        try:
            with open(self.state_path, "w") as f:
                data = self.data.model_dump()
                json.dump(data, f, indent=2)
        except Exception:
            pass

    def extract_from(self, msgs):
        if not msgs:
            return

        for max_msgs in [len(msgs), 6, 3]:
            convo_parts = []
            for m in msgs[:max_msgs]:
                role = m["role"]
                content = m["content"][:300]
                convo_parts.append(f"{role}: {content}")
                
            convo = "\n".join(convo_parts)
            
            try:
                state_json = json.dumps(self.data.model_dump(), indent=2)
                result = self.chain.invoke({
                    "state": state_json,
                    "convo": convo
                })

                merged_facts = list(dict.fromkeys(self.data.key_facts + result.key_facts))
                merged_entities = list(dict.fromkeys(self.data.core_entities + result.core_entities))
                merged_prefs = list(dict.fromkeys(self.data.user_preferences + result.user_preferences))

                topic = result.current_topic
                if not topic:
                    topic = self.data.current_topic

                self.data = MemorySchema(
                    current_topic=topic,
                    core_entities=merged_entities,
                    key_facts=merged_facts,
                    user_preferences=merged_prefs,
                )
                
                self.save()
                break

            except Exception as e:
                logger.warning("ContextGC extraction failed (max_msgs=%d): %s", max_msgs, e)
                if max_msgs == 3:
                    logger.warning("ContextGC: all retries exhausted")

    def as_system_message(self):
        has_content = self.data.current_topic or self.data.key_facts
        if not has_content:
            return None

        lines = ["[Memory] Persistent context from this conversation:"]
        
        if self.data.current_topic:
            lines.append(f"Topic: {self.data.current_topic}")
            
        if self.data.core_entities:
            entities_str = ", ".join(self.data.core_entities)
            lines.append(f"Entities: {entities_str}")
            
        if self.data.key_facts:
            lines.append("Key facts:")
            for fact in self.data.key_facts:
                lines.append(f"  - {fact}")
                
        if self.data.user_preferences:
            prefs_str = ", ".join(self.data.user_preferences)
            lines.append(f"Preferences: {prefs_str}")
            
        return "\n".join(lines)

    def is_empty(self):
        has_topic = bool(self.data.current_topic)
        has_facts = bool(self.data.key_facts)
        return not (has_topic or has_facts)
