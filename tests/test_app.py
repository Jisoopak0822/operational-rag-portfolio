from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_health():
    response = client.get('/health')
    assert response.status_code == 200
    assert response.json()['indexed_chunks'] > 0


def test_chat_vpn():
    response = client.post('/chat', json={'message': 'VPN 오류 720 해결 방법'})
    assert response.status_code == 200
    data = response.json()
    assert data['sources'][0]['source'] == 'vpn_guide.md'
    assert data['mode'] in ('retrieval_only', 'llm', 'fallback')


def test_openai_compatible():
    response = client.post('/v1/chat/completions', json={
        'messages': [{'role': 'user', 'content': '계정 비밀번호 재설정'}]
    })
    assert response.status_code == 200
    assert response.json()['choices'][0]['message']['role'] == 'assistant'


def test_streaming_unsupported():
    response = client.post('/v1/chat/completions', json={
        'messages': [{'role': 'user', 'content': 'VPN'}], 'stream': True
    })
    assert response.status_code == 400
