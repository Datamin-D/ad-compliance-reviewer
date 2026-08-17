"""
검색(Retrieval) — 파이프라인 [2]단계.
광고 claim을 질의로 관련 규정 top-k를 가져온다.

────────────────────────────────────────────────────────────────
ChromaDB 5분 요약 (처음이면 여기부터)
────────────────────────────────────────────────────────────────
Chroma는 "벡터를 넣고 비슷한 걸 찾는" 것만 하는 작은 DB다.
SQL DB에 비유하면:

    chromadb.PersistentClient(path)  ≈  DB 접속        (서버 없음. 그냥 폴더)
    client.get_or_create_collection  ≈  CREATE TABLE IF NOT EXISTS
    collection.upsert(...)           ≈  INSERT ... ON DUPLICATE KEY UPDATE
    collection.query(...)            ≈  SELECT ... ORDER BY 거리 LIMIT k

한 행(row)에 들어가는 것은 4가지다.

    ids        필수. 기본키. 우리는 "R-01" 같은 규정 ID를 그대로 쓴다.
    embeddings 벡터. 유사도 계산에 실제로 쓰이는 유일한 값.
    documents  원본 텍스트. 검색에 쓰이진 않고 사람이 확인할 때만 씀.
    metadatas  부가 정보(옵션). 우리는 안 씀 — 규정 원본을 파이썬에 이미 들고 있어서.

알아둘 것 3가지
1. upsert는 id가 같으면 덮어쓴다. → build를 몇 번 돌려도 중복이 안 쌓인다.
2. 거리 함수는 컬렉션을 "만들 때" metadata={"hnsw:space": "cosine"}로 정한다.
   기본값이 L2(유클리드)라서 명시 안 하면 코사인이 아니다. 나중에 못 바꾼다.
3. query()가 돌려주는 값은 한 겹 더 감싸여 있다:
       {"ids": [["R-01", "R-04"]], "distances": [[0.11, 0.23]]}
   질의를 여러 개 한 번에 넣을 수 있어서 바깥 리스트가 "질의별"이다.
   우리는 질의 1개만 넣으므로 항상 [0]을 꺼낸다.

거리 → 점수
   cosine space에서 distance = 1 - 코사인유사도. 그래서 score = 1 - distance.
   score 1.0 = 완전 동일, 0.0 = 무관.

왜 Chroma를 쓰나 (규정 28개면 numpy 3줄로도 되는데)
   실제 업무 규모에서 쓰게 될 물건이라서. 지금 규모에선 오버스펙이 맞다.
"""

from __future__ import annotations

import chromadb

from src import config
from src.schema import Policy, RetrievedPolicy, load_policies


class PolicyRetriever:
    """규정 벡터 인덱스를 만들고(build) 검색한다(search)."""

    def __init__(self, embed_fn=None, policies=None, path=None):
        """
        Args:
            embed_fn: (texts, task_type) -> 벡터 리스트.
                      기본값은 Gemini. 테스트에서는 API를 안 타는 stub을 넣는다.
            policies: 규정 목록. None이면 data/policies.jsonl에서 읽는다.
            path:     Chroma 저장 폴더. None이면 .chroma/
        """
        if embed_fn is None:
            from src import llm  # 지연 import: 테스트가 google-genai를 안 건드리게

            embed_fn = llm.embed_texts
        self._embed = embed_fn

        # rule_id -> Policy. 검색은 ID만 돌려주므로 원본 규정으로 되돌릴 표가 필요하다.
        self._policies = {p.rule_id: p for p in (policies or load_policies(config.POLICIES_DIR))}

        client = chromadb.PersistentClient(path=str(path or config.CHROMA_PATH))
        self._col = client.get_or_create_collection(
            name=config.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},  # 위 주석 2번 참고. 생성 시에만 유효
        )

    def build(self) -> int:
        """규정 전체를 임베딩해서 Chroma에 넣는다. 넣은 개수 반환."""
        policies = list(self._policies.values())
        docs = [p.to_document() for p in policies]

        # task_type="RETRIEVAL_DOCUMENT" — 색인용 임베딩.
        # 질의용과 다른 벡터가 나온다(비대칭 임베딩). 검색 품질이 눈에 띄게 좋아진다.
        vectors = self._embed(docs, "RETRIEVAL_DOCUMENT")

        self._col.upsert(
            ids=[p.rule_id for p in policies],
            embeddings=vectors,
            documents=docs,
            # 각 규정에 국가 태그를 붙여둔다. 검색 때 이 태그로 걸러 국가를 분리한다.
            metadatas=[{"country": p.country} for p in policies],
        )
        return len(policies)

    def search(self, query, k=config.TOP_K, country=None) -> list[RetrievedPolicy]:
        """
        질의 문자열 → 관련 규정 top-k (유사도 내림차순).

        country를 주면 그 국가 규정만 검색한다(Chroma의 where 필터).
        None이면 국가 구분 없이 전체에서 검색.
        """
        vector = self._embed([query], "RETRIEVAL_QUERY")[0]
        where = {"country": country} if country else None
        res = self._col.query(query_embeddings=[vector], n_results=k, where=where)

        hits = []
        # [0] = 첫 번째(이자 유일한) 질의의 결과. 위 주석 3번 참고.
        for rule_id, dist in zip(res["ids"][0], res["distances"][0]):
            policy = self._policies.get(rule_id)
            if policy:  # 규정 파일에서 지웠는데 재색인을 안 한 ID는 건너뜀
                hits.append(RetrievedPolicy(policy, 1.0 - float(dist)))
        return hits

    def search_many(self, queries, k=config.TOP_K, country=None) -> list[RetrievedPolicy]:
        """
        claim 여러 개로 검색한 뒤 중복 규정을 합친다(같은 규정은 최고 점수만).
        claim 3개 × k=5 = 15건을 그대로 프롬프트에 넣으면 중복으로 토큰만 낭비된다.
        """
        best: dict[str, RetrievedPolicy] = {}
        for q in queries:
            for hit in self.search(q, k, country=country):
                rid = hit.policy.rule_id
                if rid not in best or hit.score > best[rid].score:
                    best[rid] = hit
        return sorted(best.values(), key=lambda h: h.score, reverse=True)

    def count(self) -> int:
        """인덱스에 들어 있는 규정 수. 0이면 아직 build를 안 한 것."""
        return self._col.count()
