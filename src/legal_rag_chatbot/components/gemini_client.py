from legal_rag_chatbot.logging import logger
from legal_rag_chatbot.entity import (
    GeminiConfig,
    RetrieveDecision,
    IsREL,
    IsSUP,
    IsUSE
)
from google import genai
from google.genai import types

class GeminiClient:
    """Gemini API 客戶端封裝"""
    
    def __init__(self, config: GeminiConfig):
        self.config = config
        self.client = genai.Client(api_key=config.gemini_api_key)
        logger.info(f"Gemini client initialized, using model: {config.gemini_model}")
            
    def generate_content(self, prompt: str, context: str = "") -> str:
        """生成文本內容"""
        try:
            full_prompt = f"{context}\n\n{prompt}" if context else prompt

            response = self.client.models.generate_content(
                model=self.config.gemini_model,
                contents=full_prompt,
                config=types.GenerateContentConfig(
                    temperature=self.config.gemini_temperature,
                    max_output_tokens=self.config.gemini_max_output_tokens
                )
            )
            
            print(f"產生內容===>: {response.candidates[0].content.parts[0].text}")
            
            if response and hasattr(response, 'text') and response.text:
                return response.text.strip()
            else:
                # 如果因為安全設定等原因沒有文字返回，則返回空字串
                logger.warning(f"Gemini API did not return text. Full response: {response}")
                return ""
            
        except Exception as e:
            logger.error(f"Gemini API 呼叫失敗: {str(e)}")
            return "抱歉，系統暫時無法處理您的請求。"
    
    def predict_retrieve(self, query: str, previous_generation: str = "") -> RetrieveDecision:
        """
        M predicts Retrieve given (x, y_{t-1})
        判斷是否需要檢索對話資料來回答問題
        """
        prompt = f"""
                    請判斷以下使用者查詢是否需要檢索對話資料庫來回答：

                    目前使用者查詢: {query}
                    之前的對話內容: {previous_generation if previous_generation else "無"}

                    判斷標準：
                    - yes: 查詢需要具體的對話內容或對話場景來回答。如果有之前的對話內容，請務必也要引用
                    - no: 查詢是一般性問題，可以直接回答，不需要特定對話內容

                    只允許回答: yes/no
                    """
        response = self.generate_content(prompt).lower().strip()
        if "yes" in response:
            return RetrieveDecision.YES
        else:
            return RetrieveDecision.NO
    
    def predict_isrel(self, query: str, dialogue: str) -> IsREL:
        """
        M predicts IsREL given x, d
        判斷對話是否與查詢相關
        """
        prompt = f"""
                請判斷以下對話內容是否與使用者查詢相關：

                使用者查詢: {query}
                對話內容: {dialogue}...

                判斷標準：
                - relevant: 對話包含與查詢直接相關的資訊、情境或主題
                - irrelevant: 對話與查詢無關或關聯性極低

                只允許回答: relevant/irrelevant
                """
        response = self.generate_content(prompt).lower().strip()
        return IsREL.RELEVANT if "relevant" in response else IsREL.IRRELEVANT
    
    def predict_issup(self, query: str, dialogue: str, candidate_answer: str) -> IsSUP:
        """
        M predicts IsSUP given x, y_t, d
        判斷對話是否支撐候選答案
        """
        prompt = f"""
                    請判斷以下對話內容是否支撐候選答案中的陳述：

                    使用者查詢: {query}
                    候選答案: {candidate_answer}
                    對話內容: {dialogue}...

                    判斷標準：
                    - fully supported: 答案中的陳述完全可在對話中找到依據
                    - partially supported: 部分陳述有依據，部分沒有
                    - no support: 答案沒有對話依據

                    只允許回答: fully supported/partially supported/no support
                """
        response = self.generate_content(prompt).lower().strip()
        if "fully supported" in response:
            return IsSUP.FULLY_SUPPORTED
        elif "partially supported" in response:
            return IsSUP.PARTIALLY_SUPPORTED
        else:
            return IsSUP.NO_SUPPORT
    
    def predict_isuse(self, query: str, candidate_answer: str, dialogue: str = "") -> IsUSE:
        """
        M predicts IsUSE given x, y_t, d
        評估候選答案的有用性
        """
        context = f"參考對話: {dialogue}..." if dialogue else ""
        prompt = f"""
                    請評估以下候選答案對使用者查詢的有用性(1-5分):

                    使用者查詢: {query}
                    候選答案: {candidate_answer}
                    {context}

                    評分標準：
                    5分 - 非常有用：完整回答問題，提供具體相關資訊
                    4分 - 有用：回答相關且有幫助
                    3分 - 中等：部分相關但不夠詳細
                    2分 - 較少用：相關性低或幫助有限
                    1分 - 無用：不相關或誤導性資訊

                    只允許回答數字: 1-5
                """
        response = self.generate_content(prompt).strip()
        try:
            score = int(response)
            return IsUSE(score) if 1 <= score <= 5 else IsUSE.MODERATELY_USEFUL
        except:
            return IsUSE.MODERATELY_USEFUL