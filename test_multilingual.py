import requests
import json

# Test multilingual prediction
url = 'http://localhost:8000/predict/multilingual'
test_texts = [
    'This is a great product!',
    'You are stupid and should die',
    'Eres muy estúpido',  # Spanish: 'You are very stupid'
    'Tu es très intelligent'  # French: 'You are very intelligent'
]

print("Testing Multilingual Endpoint:")
print("=" * 40)

for i, text in enumerate(test_texts, 1):
    try:
        response = requests.post(url, json={'text': text})
        if response.status_code == 200:
            result = response.json()
            print(f'Test {i}: {text}')
            print(f'  Prediction: {result["prediction"]}')
            print(f'  Confidence: {result["confidence"]:.1%}')
            print(f'  Language: {result.get("detected_language", "N/A")}')
            print(f'  Processing Time: {result.get("processing_time_ms", "N/A")}ms')
            print()
        else:
            print(f'Test {i} failed: {response.status_code} - {response.text}')
    except Exception as e:
        print(f'Test {i} error: {e}')

print("✅ Multilingual tests completed!")