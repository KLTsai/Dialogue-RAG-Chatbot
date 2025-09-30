from legal_rag_chatbot.logging import logger
from legal_rag_chatbot.configuration import ConfigurationManager
from legal_rag_chatbot.rag_sys import DialogueRAGSystem
import os
import time

class MainChatbotInterface:
    def __init__(self):
        """在應用程式啟動時完成初始化"""
        logger.info("🚀 開始初始化對話RAG系統...")
        self.rag_system = None
        self.config = None
        self.initialization_message = ""

        self._startup_initialization()

    def _startup_initialization(self):
        """應用程式啟動時的初始化流程"""
        try:
            start_time = time.time()

            # 檢查環境變數
            api_key = os.getenv('GEMINI_API_KEY')
            if not api_key:
                self.initialization_message = "❌ 未找到 GEMINI_API_KEY 環境變數。請在 Hugging Face Space 的 Settings 中設置 API 密鑰。"
                logger.error("缺少必要的 API 密鑰環境變數")
                return

            logger.info("✅ API 密鑰檢查通過，開始載入配置...")

            # 初始化配置
            self.config = ConfigurationManager()
            logger.info("✅ 配置載入完成，開始初始化 RAG 系統...")

            # 初始化 RAG 系統
            self.rag_system = DialogueRAGSystem(self.config)
            logger.info("✅ RAG 系統創建完成，開始建立知識庫...")

            # 建立知識庫
            self.rag_system.build_knowledge_base()

            init_time = time.time() - start_time
            self.initialization_message = f"✅ 系統初始化完成！耗時 {init_time:.2f} 秒"
            logger.info(f"🎉 對話RAG系統初始化成功，耗時 {init_time:.2f} 秒")

        except Exception as e:
            error_msg = f"初始化失敗: {str(e)}"
            self.initialization_message = f"❌ {error_msg}"
            logger.error(f"RAG系統初始化失敗: {str(e)}")
            self.rag_system = None

    def chat_response(self, message, history):
        """處理聊天回應"""
        # 檢查系統是否已初始化
        if not self.rag_system:
            return f"❌ 系統尚未初始化或初始化失敗。\n{self.initialization_message}\n\n請檢查 API 密鑰設置或聯繫管理員。"

        try:
            # 轉換歷史對話格式
            conversation_history = []
            if history:
                for conv in history:
                    if conv.get('role') == 'user':
                        conversation_history.append(f"User: {conv.get('content', '')}")
                    elif conv.get('role') == 'assistant':
                        conversation_history.append(f"Assistant: {conv.get('content', '')}")

            # 查詢 RAG 系統
            result = self.rag_system.query(message, conversation_history)

            # 格式化回應
            response = result['answer']

            # 添加額外資訊
            metadata_parts = []
            if result.get('sources'):
                metadata_parts.append(f"📚 參考來源: {', '.join(result['sources'])}")

            if result.get('support_level'):
                metadata_parts.append(f"🎯 支持程度: {result['support_level']}")

            if result.get('usefulness_score'):
                metadata_parts.append(f"⭐ 有用性評分: {result['usefulness_score']}/5")

            if result.get('processing_time'):
                metadata_parts.append(f"⏱️ 處理時間: {result['processing_time']:.2f}秒")

            if metadata_parts:
                response += "\n\n" + "\n".join(metadata_parts)

            return response

        except Exception as e:
            logger.error(f"查詢處理錯誤: {str(e)}")
            return f"❌ 處理查詢時發生錯誤: {str(e)}"

    def get_system_status(self):
        """獲取系統狀態資訊"""
        if self.rag_system:
            stats = self.rag_system.get_system_stats()
            status_info = []
            status_info.append(f"✅ 系統狀態: 正常運行")
            status_info.append(f"📊 總查詢數: {stats.get('total_queries', 0)}")
            status_info.append(f"🔍 檢索查詢數: {stats.get('retrieval_queries', 0)}")
            status_info.append(f"💬 非檢索查詢數: {stats.get('non_retrieval_queries', 0)}")
            status_info.append(f"📚 知識庫對話數: {stats.get('dialogues_in_kb', 0)}")
            status_info.append(f"⏰ 運行時間: {stats.get('uptime_hours', 0):.2f} 小時")
            return "\n".join(status_info)
        else:
            return f"❌ 系統未初始化\n{self.initialization_message}"