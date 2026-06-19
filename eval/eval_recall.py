import time
import os
from contextgc.core.eviction import EvictionOrchestrator

FACTS = [
    {"statement": "I'm thinking of moving to Seattle next year.", "query": "Which Pacific Northwest city am I considering relocating to?", "expected": "Seattle"},
    {"statement": "My favorite programming language is Rust because of its safety.", "query": "Which backend tech do I prefer due to memory guarantees?", "expected": "Rust"},
    {"statement": "I just bought a new mechanical keyboard with brown switches.", "query": "What type of tactile hardware did I purchase for typing?", "expected": "brown"},
    {"statement": "I have an allergy to peanuts.", "query": "Which legume causes an adverse immune reaction for me?", "expected": "peanuts"},
    {"statement": "My best friend from high school is named Alex.", "query": "Who was my closest companion during my teenage education?", "expected": "Alex"},
    {"statement": "I am currently reading Dune by Frank Herbert.", "query": "Which sci-fi desert novel am I presently focused on?", "expected": "Dune"},
    {"statement": "For breakfast today I had avocado toast and black coffee.", "query": "What green fruit did I consume this morning?", "expected": "avocado"},
    {"statement": "I drive a 2018 Honda Civic.", "query": "What Japanese sedan do I commute in?", "expected": "Civic"},
    {"statement": "My goal for this year is to run a marathon in under 4 hours.", "query": "What long-distance athletic milestone am I training for?", "expected": "marathon"},
    {"statement": "I graduated from university in 2022.", "query": "In which calendar year did I complete my higher education degree?", "expected": "2022"}
]

DISTRACTORS = [
    "I visited Portland last year, it's a great city for moving to.",
    "C++ is a powerful language but manual memory management is hard.",
    "I have a blue membrane keyboard at the office.",
    "I am not allergic to tree nuts or dairy.",
    "My college roommate was named Sam.",
    "I watched the movie adaptation of Foundation recently.",
    "I drank green tea and ate eggs for lunch.",
    "My brother drives a Toyota Corolla.",
    "I want to do a 5k run next month.",
    "I started university in 2018."
]

def generate_conversation(num_turns, facts):
    messages = []
    
    fact_interval = max(1, num_turns // len(facts))
    fact_idx = 0
    
    distractor_interval = max(1, num_turns // len(DISTRACTORS))
    distractor_idx = 0

    for i in range(num_turns):
        if i % fact_interval == 0 and fact_idx < len(facts):
            content = facts[fact_idx]["statement"]
            fact_idx += 1
        elif i % distractor_interval == 0 and distractor_idx < len(DISTRACTORS):
            content = DISTRACTORS[distractor_idx]
            distractor_idx += 1
        else:
            content = f"Can you tell me more about general topic {i}?"
            
        messages.append({
            "id": f"msg_user_{i}",
            "role": "user",
            "content": content,
            "timestamp": time.time()
        })
        messages.append({
            "id": f"msg_asst_{i}",
            "role": "assistant",
            "content": f"Sure, here is some information regarding your request.",
            "timestamp": time.time()
        })
        
    return messages

def evaluate_recall(num_turns):
    print(f"running {num_turns} turns...")
    
    db_path = f"test_recall_{num_turns}.json"
    if os.path.exists(db_path):
        os.remove(db_path)
        
    gc = EvictionOrchestrator(model="qwen2.5", max_tokens=200, watermark=0.5, state_path=db_path)
    
    messages = generate_conversation(num_turns, FACTS)
    
    gc.process(messages)
    
    if not gc.archive:
        print("No messages were archived! Something went wrong.")
        return
        
    hits_at_1 = 0
    hits_at_3 = 0
    mrr_sum = 0.0
    
    for fact in FACTS:
        query = fact["query"]
        expected = fact["expected"]
        
        results = gc.archive.recall_relevant_messages(query, threshold=0.1)
        
        rank = 0
        for i, (res_dict, score) in enumerate(results[:3]):
            if expected.lower() in res_dict["content"].lower():
                rank = i + 1
                break
                
        if rank == 1:
            hits_at_1 += 1
            hits_at_3 += 1
            mrr_sum += 1.0
        elif rank > 1:
            hits_at_3 += 1
            mrr_sum += 1.0 / rank
            
    total = len(FACTS)
    recall_1 = (hits_at_1 / total) * 100
    recall_3 = (hits_at_3 / total) * 100
    mrr = mrr_sum / total
    
    print(f"r1: {recall_1:.1f}% | r3: {recall_3:.1f}% | mrr: {mrr:.3f}")
    
    return {"turns": num_turns, "r1": recall_1, "r3": recall_3, "mrr": mrr}

def main():
    results = []
    for turns in [50, 100]:
        res = evaluate_recall(turns)
        if res:
            results.append(res)
            
if __name__ == "__main__":
    main()
