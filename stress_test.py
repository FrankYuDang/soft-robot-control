import time
import requests
import json
import concurrent.futures

# API 地址
URL = "http://127.0.0.1:8000/predict"

# 模拟数据
PAYLOAD = {
  "sequence": [[0.5]*3 for _ in range(10)]
}

def send_request(request_id):
    try:
        start = time.time()
        response = requests.post(URL, json=PAYLOAD)
        latency = (time.time() - start) * 1000
        return latency, response.status_code
    except Exception as e:
        return 0, 500

def run_stress_test(total_requests=100, concurrency=10):
    print(f"🚀 Starting Stress Test: {total_requests} requests, {concurrency} threads...")
    
    latencies = []
    success_count = 0
    
    # 使用线程池并发发送请求
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(send_request, i) for i in range(total_requests)]
        
        for future in concurrent.futures.as_completed(futures):
            lat, status = future.result()
            if status == 200:
                success_count += 1
                latencies.append(lat)

    # 统计结果
    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    max_latency = max(latencies) if latencies else 0
    
    print("-" * 30)
    print(f"✅ Success Rate: {success_count}/{total_requests}")
    print(f"⏱️  Average Latency: {avg_latency:.2f} ms")
    print(f"🐢 Max Latency:     {max_latency:.2f} ms")
    print("-" * 30)

if __name__ == "__main__":
    # 先跑一次热身
    run_stress_test(total_requests=10, concurrency=1)
    # 再跑并发测试
    run_stress_test(total_requests=200, concurrency=20)