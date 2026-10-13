"""Run: python -m uvicorn main:app --reload --port 8000"""
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from config import ROOT, settings
from core.qa_service import QAService
from ingest.pipeline import load_knowledge_base
from rag.agent import RAGAgent
from rag.generator import RAGGenerator
from rag.retriever_v2 import HybridRetriever


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=5000)
    top_k: int = Field(default=5, ge=1, le=25)


class CompletionMessage(BaseModel):
    role: str
    content: str


class CompletionRequest(BaseModel):
    model: str | None = None
    messages: list[CompletionMessage]
    stream: bool = False


def create_service() -> QAService:
    chunks = load_knowledge_base(settings.docs_dir)
    retriever = HybridRetriever(chunks, backend=settings.embedding_backend,
                                model_name=settings.embedding_model)
    generator = RAGGenerator(api_key=settings.llm_api_key,
                             base_url=settings.llm_base_url,
                             model=settings.llm_model, timeout_s=settings.llm_timeout_s)
    return QAService(RAGAgent(retriever), generator, settings.top_k)


app = FastAPI(title='Operational RAG Portfolio (Stage 1)', version='0.1.0')
service = create_service()


@app.get('/')
def home():
    return FileResponse(ROOT / 'web' / 'index.html')


@app.get('/health')
def health():
    return {
        'ok': True,
        'ready': bool(service.agent.retriever.chunks),
        'indexed_chunks': len(service.agent.retriever.chunks),
        'embedding_backend': service.agent.retriever.backend,
        'llm_configured': bool(service.generator.api_key),
    }


@app.get('/metrics')
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post('/chat')
def chat(request: ChatRequest):
    return service.process_query(request.message, request.top_k)


@app.post('/v1/chat/completions')
def chat_completions(request: CompletionRequest):
    if request.stream:
        raise HTTPException(status_code=400, detail='Streaming is not supported in Stage 1')
    user_messages = [m.content for m in request.messages if m.role == 'user']
    if not user_messages or not user_messages[-1].strip():
        raise HTTPException(status_code=400, detail='A nonempty user message is required')
    result = service.process_query(user_messages[-1])
    return {
        'id': f'chatcmpl-local-{int(time.time())}',
        'object': 'chat.completion',
        'created': int(time.time()),
        'model': request.model or 'local-rag',
        'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': result['answer']},
                     'finish_reason': 'stop'}],
        'sources': result['sources'],
        'rag_mode': result['mode'],
    }
