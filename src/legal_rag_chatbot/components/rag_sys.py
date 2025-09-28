import time
from datetime import datetime
from datasets import load_from_disk
from typing import List, Dict, Any, Tuple
import numpy as np
from legal_rag_chatbot.logging import logger
from legal_rag_chatbot.config.configuration import ConfigurationManager
from legal_rag_chatbot.entity import (
    GeminiConfig,
    RetrieveDecision,
    IsREL,
    IsSUP,
    IsUSE
)
from legal_rag_chatbot.components.data_embedding import EmbeddingModel
from legal_rag_chatbot.components.vector_storage import VectorStorage
from legal_rag_chatbot.components.gemini_client import GeminiClient

class DialogueRAGSystem:
    """基於 SAMSum 對話資料集的 RAG 系統"""

    def __init__(self, config: ConfigurationManager):
        self.config = config
        self.embedding_model_config = self.config.get_data_embedding_config()
        self.embedding_model = EmbeddingModel(config= self.embedding_model_config)
        
        self.vector_storage_config = self.config.get_vector_storage_config()
        self.gemini_config = self.config.get_gemini_config()
        self.rag_sys_config = self.config.get_rag_sys_config()

        self.vector_store = VectorStorage(config=self.vector_storage_config)
        self.dataset_with_embeddings = load_from_disk(self.vector_storage_config.data_path)
        self.gemini_client = GeminiClient(config=self.gemini_config)
        self.config.max_retrieval_docs = self.rag_sys_config.max_retrieval_docs
        
        # 系統統計
        self.stats = {
            "total_queries": 0,
            "retrieval_queries": 0,
            "non_retrieval_queries": 0,
            "start_time": datetime.now()
        }

        logger.info("Dialogue RAG System 初始化完成")

    def build_knowledge_base(self):
        """建立對話知識庫"""
        logger.info("開始建立對話知識庫...")

        # 載入和處理對話
        documents = [{"id": item["id"], "dialogue": item["dialogue"], "summary": item["summary"]} for item in self.dataset_with_embeddings]
        embeddings = np.array(self.dataset_with_embeddings["embedding"])
        
        # 添加到向量存儲
        self.vector_store.add_documents(documents, embeddings)

    def query(self, user_query: str, conversation_history: List[str] = None) -> Dict[str, Any]:
        """
        主要查詢函數 - 實作完整的對話檢索流程
        """
        start_time = time.time()
        self.stats["total_queries"] += 1

        # Step 1: M predicts Retrieve given (x, y_{t-1})
        previous_generation = "\n".join(conversation_history[-3:]) if conversation_history else ""
        retrieve_decision = self.gemini_client.predict_retrieve(user_query, previous_generation)
        

        if retrieve_decision == RetrieveDecision.YES:
            return self._handle_retrieval_branch(user_query, previous_generation, start_time)
        else:
            return self._handle_non_retrieval_branch(user_query, start_time)

    def _handle_retrieval_branch(self, query: str, previous_generation: str, start_time: float) -> Dict[str, Any]:
        """處理需要檢索的分支"""
        self.stats["retrieval_queries"] += 1

        # Step 4: 檢索相關對話
        query_embedding = self.embedding_model.encode([query])[0]
        retrieved_docs = self.vector_store.similarity_search(
            query_embedding,
            top_k=self.rag_sys_config.max_retrieval_docs
        )

        if not retrieved_docs:
            return {
                "answer": "抱歉，我找不到相關的對話內容來回答您的問題。",
                "retrieve_decision": RetrieveDecision.YES.value,
                "sources": [],
                "processing_time": time.time() - start_time
            }

        # Step 5-7: 為每個相關對話生成候選答案並評估
        candidates = []

        for doc, score in retrieved_docs:
            # 判斷相關性
            
            relevance = self.gemini_client.predict_isrel(query, doc['dialogue'])
            # print(f"predict_isrel ====> {relevance}")
            if relevance == IsREL.RELEVANT:
                # 生成候選答案
                candidate_answer = self._generate_candidate_answer(query, doc, previous_generation)

                # 評估支撐性和有用性
                support_level = self.gemini_client.predict_issup(query, doc['dialogue'], candidate_answer)
                usefulness = self.gemini_client.predict_isuse(query, candidate_answer, doc['dialogue'])

                candidates.append({
                    'answer': candidate_answer,
                    'source_doc': doc,
                    'is_relevant': relevance,
                    'support_level': support_level,
                    'usefulness_score': usefulness
                })

        if not candidates:
            return {
                "answer": "檢索到的對話內容與您的問題不太相關，無法提供有效回答。",
                "retrieve_decision": RetrieveDecision.YES.value,
                "sources": [doc['id'] for doc, score in retrieved_docs],
                "processing_time": time.time() - start_time
            }

        # Step 8: 選擇最佳候選答案
        best_candidate = self._rank_candidates(candidates)

        return {
            "answer": best_candidate['answer'],
            "retrieve_decision": RetrieveDecision.YES.value,
            "sources": [best_candidate['source_doc']['id']],
            "relevance": best_candidate['is_relevant'].value,
            "support_level": best_candidate['support_level'].value,
            "usefulness_score": best_candidate['usefulness_score'].value,
            "processing_time": time.time() - start_time,
            "reference_dialogue": best_candidate['source_doc']['dialogue'][:300] + "..."
        }

    def _handle_non_retrieval_branch(self, query: str, start_time: float) -> Dict[str, Any]:
        """處理不需要檢索的分支"""
        self.stats["non_retrieval_queries"] += 1

        # Step 9: 直接生成答案
        generated_answer = self._generate_direct_answer(query)

        # Step 10: 評估有用性
        usefulness_score = self.gemini_client.predict_isuse(query, generated_answer)

        return {
            "answer": generated_answer,
            "retrieve_decision": RetrieveDecision.NO.value,
            "sources": [],
            "usefulness_score": usefulness_score.value,
            "processing_time": time.time() - start_time
        }

    def _generate_candidate_answer(self, query: str, dialogue_doc: Dict[str, Any], previous_generation: str) -> str:
        """為特定對話生成候選答案"""
        context = f"請基於以下對話內容回答使用者問題：\n\n對話內容: {dialogue_doc['dialogue']}\n對話摘要: {dialogue_doc['summary']}"

        if previous_generation:
            context += f"\n\n之前的對話: {previous_generation}"

        prompt = f"""
                    {context}

                    使用者問題: {query}

                    請提供有用的回答，並：
                    1. 直接回答使用者問題
                    2. 引用或描述相關的對話內容
                    3. 提供具體的資訊或見解
                    4. 保持回答簡潔明確
                """
        return self.gemini_client.generate_content(prompt)

    def _generate_direct_answer(self, query: str) -> str:
        """直接生成答案（不使用檢索）"""
        prompt = f"""
                    作為對話理解助手，請回答以下問題：

                    {query}

                    請基於一般知識提供回答，並：
                    1. 直接回答問題
                    2. 提供相關的背景資訊
                    3. 保持回答有用且相關
                """
        return self.gemini_client.generate_content(prompt)

    def _rank_candidates(self, candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """根據 IsREL, IsSUP, IsUSE 排序候選答案"""
        def calculate_score(candidate):
            score = 0

            # IsREL 權重
            if candidate['is_relevant'] == IsREL.RELEVANT:
                score += 10

            # IsSUP 權重
            support_scores = {
                IsSUP.FULLY_SUPPORTED: 10,
                IsSUP.PARTIALLY_SUPPORTED: 5,
                IsSUP.NO_SUPPORT: 0
            }
            score += support_scores.get(candidate['support_level'], 0)

            # IsUSE 權重
            score += candidate['usefulness_score'].value * 2

            return score

        # 按分數排序
        sorted_candidates = sorted(candidates, key=calculate_score, reverse=True)
        return sorted_candidates[0]

    def get_system_stats(self) -> Dict[str, Any]:
        """獲取系統統計資訊"""
        uptime = datetime.now() - self.stats["start_time"]
        return {
            "total_queries": self.stats["total_queries"],
            "retrieval_queries": self.stats["retrieval_queries"],
            "non_retrieval_queries": self.stats["non_retrieval_queries"],
            "retrieval_rate": self.stats["retrieval_queries"] / max(1, self.stats["total_queries"]),
            "uptime_hours": uptime.total_seconds() / 3600,
            "dialogues_in_kb": len(self.vector_store.documents)
        }