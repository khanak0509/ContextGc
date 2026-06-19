import time
from contextgc.core.eviction import EvictionOrchestrator

def run_benchmark():
    print("Initializing Benchmark...")
    # Initialize with a very small token limit to force heavy eviction
    gc = EvictionOrchestrator(model="qwen2.5", max_tokens=500, watermark=0.5)
    
    # Create 50 user-assistant turns (100 messages total)
    messages = []
    print("Generating 100 dummy messages...")
    for i in range(50):
        messages.append({
            "id": f"msg_user_{i}",
            "role": "user",
            "content": f"Hello, my favorite color is color_{i} and my favorite animal is animal_{i}.",
            "timestamp": time.time()
        })
        messages.append({
            "id": f"msg_asst_{i}",
            "role": "assistant",
            "content": f"That's great to know that you like color_{i} and animal_{i}!",
            "timestamp": time.time()
        })
        
    print(f"Total initial messages: {len(messages)}")
    
    # Benchmark 1: Eviction and Compression Time
    print("--- Running Eviction/Compression Benchmark ---")
    start_time = time.time()
    compressed_messages = gc.process(messages)
    end_time = time.time()
    
    eviction_time = end_time - start_time
    print(f"Time to process and compress: {eviction_time:.2f} seconds")
    print(f"Total messages after compression: {len(compressed_messages)}")
    print(f"Total evictions performed: {gc.total_evictions}")
    
    # Add a new message that should trigger a recall
    print("--- Running Hybrid Recall Benchmark ---")
    recall_query = [
        {"id": "msg_recall_1", "role": "user", "content": "What was my favorite color at turn 10?", "timestamp": time.time()}
    ]
    
    start_time = time.time()
    # Process just the recall query (should trigger BM25 + Vector search)
    final_messages = gc.process(compressed_messages + recall_query)
    end_time = time.time()
    
    recall_time = end_time - start_time
    print(f"Time to perform Hybrid Recall (BM25 + Vector): {recall_time:.2f} seconds")
    print(f"Total recalls performed: {gc.total_recalls}")
    
    return eviction_time, recall_time

if __name__ == "__main__":
    run_benchmark()
