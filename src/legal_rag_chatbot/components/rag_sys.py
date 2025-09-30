import time
from datetime import datetime
from datasets import load_from_disk
from typing import List, Dict, Any, Tuple
import numpy as np
from legal_rag_chatbot.utils.common import calculate_score
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
    """
    RAG System based on SAMSum dataset
    """
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
        """
        Build dialogue knowledgebase
        """
        logger.info("Building up dialogue knowledgebase...")

        # 載入和處理對話
        documents = [{"id": item["id"], "dialogue": item["dialogue"], "summary": item["summary"]} for item in self.dataset_with_embeddings]
        embeddings = np.array(self.dataset_with_embeddings["embedding"])
        
        # 添加到向量存儲
        self.vector_store.add_documents(documents, embeddings)

    def query(self, user_query: str, conversation_history: List[str] = None) -> Dict[str, Any]:
        """
        Main query function for retrieve wokflow
        """
        start_time = time.time()
        self.stats["total_queries"] += 1

        # Step 1: M predicts Retrieve given (x, y_{t-1})
        previous_generation = "\n".join(conversation_history[-3:]) if conversation_history else ""
        retrieve_decision = self.gemini_client.predict_retrieve(user_query, previous_generation)
        logger.info("Step 2: M predicts Retrieve given (x, y_{t-1})")

        if retrieve_decision == RetrieveDecision.YES:
            return self._handle_retrieval_branch(user_query, previous_generation, start_time)
        else:
            return self._handle_non_retrieval_branch(user_query, start_time)

    def _handle_retrieval_branch(self, query: str, previous_generation: str, start_time: float) -> Dict[str, Any]:
        """
        Handle retrieval branch from retrieve decision
        """
        self.stats["retrieval_queries"] += 1

        # Step 4: 檢索相關對話
        logger.info("Step 4: Retrieve relevant text passages by similarity search")
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
        logger.info("Step 5-7: Generate candidate answers for each passages and evaluate")
        candidates = []
        doc_cnt = 1
        for doc, score in retrieved_docs:
            # 判斷相關性
            doc_cnt += 1
            logger.info(f"Step 5: Predict [{doc_cnt}] retrieved_doc whether it is relevant or not")

            relevance = self.gemini_client.predict_isrel(query, doc['dialogue'])
            
            if relevance == IsREL.RELEVANT:
                # 生成候選答案
                logger.info(f"Step 5-1: [{doc_cnt}] generate candidate answer for evaluations")
                candidate_answer = self._generate_candidate_answer(query, doc, previous_generation)

                # 評估支撐性和有用性
                logger.info(f"Step 6: [{doc_cnt}] Evaluations by two metrics (IsSup & IsUse)")
                support_level = self.gemini_client.predict_issup(query, doc['dialogue'], candidate_answer)
                usefulness = self.gemini_client.predict_isuse(query, candidate_answer, doc['dialogue'])

                candidates.append({
                    'answer': candidate_answer,
                    'source_doc': doc,
                    'is_relevant': relevance,
                    'support_level': support_level,
                    'usefulness_score': usefulness
                })

        logger.info(f"Number of candidate answers: {len(candidates)}")
        
        if not candidates:
            return {
                "answer": "檢索到的對話內容與您的問題不太相關，無法提供有效回答。",
                "retrieve_decision": RetrieveDecision.YES.value,
                "sources": [doc['id'] for doc, score in retrieved_docs],
                "processing_time": time.time() - start_time
            }

        # Step 7: 選擇最佳候選答案
        logger.info(f"Step 7: Ranking candidate answers based on IsRel, IsSup, and IsUse")
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
        """
        Handle non-retrieval branch
        """
        self.stats["non_retrieval_queries"] += 1

        # Step 9: 直接生成答案
        logger.info(f"Step 9: Generate answers directly")
        generated_answer = self._generate_direct_answer(query)

        # Step 10: 評估有用性
        logger.info(f"Step 10: Evaluate answers by IsUse")
        usefulness_score = self.gemini_client.predict_isuse(query, generated_answer)

        return {
            "answer": generated_answer,
            "retrieve_decision": RetrieveDecision.NO.value,
            "sources": [],
            "usefulness_score": usefulness_score.value,
            "processing_time": time.time() - start_time
        }

    def _generate_candidate_answer(self, query: str, dialogue_doc: Dict[str, Any], previous_generation: str) -> str:
        """
        generate candidate answer for specific
        """
        
        # 結構化上下文，讓模型清楚區分不同訊息來源
        context = f"""
                    ### 對話全文
                    {dialogue_doc['dialogue']}

                    ### 對話摘要
                    {dialogue_doc['summary']}
                    """
        if previous_generation:
            context += f"\n### 之前的對話\n{previous_generation}"

        prompt = f"""
                    你是一位專業的對話分析助理。你的任務是嚴格根據dialogue knowledgebase內的「對話全文」和「對話摘要」來回答「使用者問題」。

                    **回答框架與思考流程 (Chain-of-Thought):**
                    1.  **理解問題**：首先，完全理解「使用者問題」的核心需求。
                    2.  **定位資訊**：在「對話全文」和「對話摘要」中查找與問題最相關的具體段落或句子。
                    3.  **綜合回答**：基於找到的資訊，組織一個清晰、直接的回答。

                    **評估材料:**
                    {context}

                    ---

                    **使用者問題:**
                    {query}

                    ---

                    **輸出要求:**
                    1.  **直接回答優先 (Answer First)**：回答的開頭必須直接、簡潔地回應問題的核心。
                    2.  **嚴格基於文本 (Grounding)**：所有回答都必須完全基於提供的「評估材料」。**絕對禁止**添加任何材料中未提及的外部知識或個人推測。
                    3.  **引用證據 (Cite Evidence)**：在提供細節時，應明確引用或描述相關的對話內容作為證據。例如，可以說「根據對話內容...」或簡短引用關鍵句子。
                    4.  **結構化輸出 (Structured Output)**:如果答案包含多個要點，請使用條列式(bullet points)來呈現，以提高可讀性。
                    5.  **語氣與風格 (Tone and Style)**：保持中立、客觀的助理語氣，專注於傳達事實。

                    請開始生成回答：
                """
        return self.gemini_client.generate_content(prompt)

    def _generate_direct_answer(self, query: str) -> str:
        """
        generate answer directly (not leverage retriever)
        """
        prompt = f"""
                    你是一位知識淵博且樂於助人的AI助理。你的任務是基於你的通用知識，為使用者提供清晰、準確且有用的回答。

                    **任務指示:**
                    1.  **理解意圖**: 仔細分析使用者問題的核心意圖。
                    2.  **提取知識**: 從你的知識庫中提取最相關的資訊。
                    3.  **組織答案**: 結構化地組織資訊，使其易於理解。

                    ---

                    **使用者問題:**
                    {query}

                    ---

                    **回答要求:**
                    1.  **直接回答優先 (Answer First)**: 在回答的開頭，直接且簡潔地回應問題的核心。
                    2.  **提供深度與廣度 (Provide Depth and Context)**: 在直接回答之後，提供相關的背景資訊、重要脈絡或有趣的細節，以增加回答的價值。
                    3.  **結構化輸出 (Structured Output)**: 如果答案包含多個部分、步驟或要點，請務必使用條列式(bullet points)或編號列表來呈現，以提高可讀性。
                    4.  **保持客觀與中立 (Maintain Objectivity)**: 除非被特別要求，否則應保持客觀中立的語氣，避免表達個人觀點或偏好。
                    5.  **定義關鍵術語 (Define Key Terms)**: 如果回答中包含專業術語，請提供簡潔的解釋。

                    請開始生成回答：
                """
        return self.gemini_client.generate_content(prompt)

    def _rank_candidates(self, candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Sort ranking candiates based on IsRel, IsSup, IsUse
        """
        # 按分數排序
        sorted_candidates = sorted(candidates, key=calculate_score, reverse=True)
        logger.info(f"Rerank for candidates: {sorted_candidates}")
        return sorted_candidates[0]

    def get_system_stats(self) -> Dict[str, Any]:
        """
        System statistics
        """
        uptime = datetime.now() - self.stats["start_time"]
        return {
            "total_queries": self.stats["total_queries"],
            "retrieval_queries": self.stats["retrieval_queries"],
            "non_retrieval_queries": self.stats["non_retrieval_queries"],
            "retrieval_rate": self.stats["retrieval_queries"] / max(1, self.stats["total_queries"]),
            "uptime_hours": uptime.total_seconds() / 3600,
            "dialogues_in_kb": len(self.vector_store.documents)
        }