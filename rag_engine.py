import os
import requests
import numpy as np
import faiss
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from typing import List, Tuple

class NewsRAG:
    def __init__(self, llm_provider, embedding_model="all-MiniLM-L6-v2"):
        print("🔧 Initializing News RAG...")
        self.llm = llm_provider
        self.embedder = SentenceTransformer(embedding_model)
        self.chunks = []
        self.metadata = []
        self.index = None
        print("✅ RAG ready!")

    def scrape_article(self, url: str) -> Tuple[str, str]:
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(r.text, "html.parser")

        title = soup.find("h1")
        title_text = title.get_text(strip=True) if title else "No title"

        paragraphs = soup.find_all("p")
        body = " ".join(
            p.get_text(strip=True)
            for p in paragraphs
            if len(p.get_text(strip=True)) > 40
        )
        print(f"🕷️ Scraped: {title_text[:60]}")
        return title_text, body

    def chunk_text(self, text: str, chunk_size=300, overlap=30) -> List[str]:
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            start += chunk_size - overlap
        return chunks

    def add_article(self, url: str):
        title, body = self.scrape_article(url)
        full_text = title + " " + body
        new_chunks = self.chunk_text(full_text)

        embeddings = self.embedder.encode(new_chunks, convert_to_numpy=True)
        dimension = embeddings.shape[1]

        if self.index is None:
            self.index = faiss.IndexFlatL2(dimension)

        self.index.add(embeddings.astype("float32"))

        for chunk in new_chunks:
            self.chunks.append(chunk)
            self.metadata.append({"title": title, "url": url})

        return title, len(new_chunks)

    def summarize(self, title: str, body: str) -> str:
        safe_body = body[:1500]  # truncate for Gemini free tier
        prompt = f"""Summarize this news article in 4-5 clear bullet points.
Focus on: who, what, when, where, and why.

Title: {title}
Article: {safe_body}

Summary:"""
        return self.llm.generate(prompt)

    def retrieve(self, question: str, top_k=3) -> List[Tuple[str, float, dict]]:
        if self.index is None or self.index.ntotal == 0:
            return []
        question_embedding = self.embedder.encode([question])
        distances, indices = self.index.search(
            question_embedding.astype("float32"), top_k
        )
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            chunk = self.chunks[idx]
            meta = self.metadata[idx]
            similarity = 1 / (1 + distance)
            results.append((chunk, similarity, meta))
        return results

    def ask(self, question: str, top_k=3) -> dict:
        retrieved = self.retrieve(question, top_k=top_k)
        if not retrieved:
            return {
                "question": question,
                "answer": "No articles loaded yet. Please add a news URL first.",
                "sources": []
            }

        # Truncate chunks to stay within Gemini free tier token limits
        context = "\n\n".join([chunk[:500] for chunk, _, _ in retrieved])
        prompt = f"""You are a helpful news assistant.
Answer the question using ONLY the news articles below.
If the answer is not found, say "I don't have enough information from the loaded articles."

News context:
{context}

Question: {question}
Answer:"""

        answer = self.llm.generate(prompt)
        return {
            "question": question,
            "answer": answer,
            "sources": retrieved
        }