import time
import os
import json
from contextgc.core.eviction import EvictionOrchestrator

TEST_CASES = [
    {"input": "My name is Khanak.", "expected": "khanak"},
    {"input": "I am currently living in New York City.", "expected": "new york"},
    {"input": "I work as a software engineer at a startup.", "expected": "software engineer"},
    {"input": "I have two pet cats named Luna and Artemis.", "expected": "cats"},
    {"input": "My favorite hobby is playing the acoustic guitar.", "expected": "guitar"},
    {"input": "I am 25 years old.", "expected": "25"},
    {"input": "I am allergic to dairy products.", "expected": "dairy"},
    {"input": "My sister is getting married next month in June.", "expected": "sister"},
    {"input": "I really love eating spicy Mexican food.", "expected": "mexican"},
    {"input": "I graduated from Stanford University.", "expected": "stanford"},
    {"input": "I am planning a trip to Japan this winter.", "expected": "japan"},
    {"input": "I speak English and a little bit of French.", "expected": "french"},
    {"input": "My primary laptop is an M2 MacBook Air.", "expected": "macbook"},
    {"input": "I watch a lot of sci-fi movies.", "expected": "sci-fi"},
    {"input": "My favorite sport to watch is Formula 1.", "expected": "formula 1"},
    {"input": "I usually drink green tea in the mornings.", "expected": "green tea"},
    {"input": "I am studying computer science.", "expected": "computer science"},
    {"input": "I want to build an AI startup.", "expected": "startup"},
    {"input": "I run 5k every single weekend.", "expected": "5k"},
    {"input": "My favorite season is autumn.", "expected": "autumn"}
]

def evaluate_extraction(model_name):
    print(f"evaluating {model_name}...")
    
    hits = 0
    total = len(TEST_CASES)
    
    for i, test in enumerate(TEST_CASES):
        db_path = f"test_extract_{model_name.replace(':', '_')}_{i}.json"
        if os.path.exists(db_path):
            os.remove(db_path)
            
        gc = EvictionOrchestrator(model=model_name, max_tokens=10, watermark=0.1, state_path=db_path)
        
        messages = [{"id": "sys", "role": "system", "content": "You are a helpful assistant.", "timestamp": time.time()}]
        
        for turn in range(50):
            if turn == 5:
                user_msg = test["input"]
            else:
                user_msg = f"Can we discuss topic number {turn} in more detail?"
                
            messages.append({"id": f"u_{turn}", "role": "user", "content": user_msg, "timestamp": time.time()})
            messages.append({"id": f"a_{turn}", "role": "assistant", "content": "I understand and will keep that in mind.", "timestamp": time.time()})
        
        gc.process(messages)
        
        state_text = ""
        if os.path.exists(db_path):
            try:
                with open(db_path, "r") as f:
                    state_data = json.load(f)
                    for val in state_data.values():
                        if isinstance(val, str):
                            state_text += val.lower() + " "
                        elif isinstance(val, list):
                            state_text += " ".join([str(v).lower() for v in val]) + " "
            except Exception:
                pass
                
        if test["expected"].lower() in state_text:
            hits += 1
            
        if os.path.exists(db_path):
            os.remove(db_path)
            
    score = (hits / total) * 100
    print(f"score for {model_name}: {score:.1f}%")
    return score

def main():
    models = ["llama3.2:3b", "qwen2.5"]
    results = {}
    
    for m in models:
        try:
            results[m] = evaluate_extraction(m)
        except Exception as e:
            print(f"error evaluating {m}: {e}")
            results[m] = 0.0

if __name__ == "__main__":
    main()
