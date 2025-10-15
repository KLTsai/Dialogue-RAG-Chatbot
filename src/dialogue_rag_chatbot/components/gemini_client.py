from dialogue_rag_chatbot.logging import logger
from dialogue_rag_chatbot.entity import (
    GeminiConfig,
    RetrieveDecision,
    IsREL,
    IsSUP,
    IsUSE
)
from google import genai
from google.genai import types

class GeminiClient:
    
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
                    你是一個智慧檢索決策專家，專門負責判斷是否需要從對話資料庫中檢索訊息來回答使用者查詢。

                    **檢索決策任務：**
                    基於查詢的複雜度、知識需求和上下文相關性，決定是否啟動對話檢索機制。

                    **評估材料：**
                    目前使用者查詢: {query}
                    之前的對話內容: {previous_generation if previous_generation else "無"}

                    **檢索需求評估框架：**

                    **知識來源分析：**
                    1. **內在知識充足性** - 查詢是否可用模型預訓練知識回答
                    2. **對話依賴性** - 查詢是否需要特定對話內容或上下文
                    3. **個人化需求** - 查詢是否涉及個人經驗或具體對話場景
                    4. **連續性需求** - 查詢是否與之前對話內容相關聯

                    **檢索決策標準：**

                    **yes (需要檢索)** - 符合以下任一條件：
                    - 查詢明確要求具體的對話內容、對話記錄或對話細節
                    - 查詢涉及個人經驗、回憶或具體的對話情境
                    - 查詢需要引用、分析或總結特定對話內容
                    - 查詢要求檢索過往討論的具體訊息或決定
                    - 存在之前對話內容且當前查詢與之相關聯
                    - 查詢包含「我們之前討論過」、「你記得」、「對話中提到」等指示詞

                    **no (無需檢索)** - 符合以下情況：
                    - 查詢是一般性知識問題，可用常識或預訓練知識回答
                    - 查詢涉及通用概念、定義、解釋或理論知識
                    - 查詢要求創建新內容而非引用現有對話
                    - 查詢是獨立問題，與對話歷史無關
                    - 查詢涉及數學計算、邏輯推理或程式編寫等通用技能

                    **決策流程：**
                    1. 識別查詢中的關鍵需求和依賴關係
                    2. 評估查詢是否指向特定對話內容或上下文
                    3. 判斷模型內在知識是否足以回答查詢
                    4. 考慮之前對話內容與當前查詢的關聯性
                    5. 基於綜合評估做出檢索決策

                    **特殊考量：**
                    - 當存在之前對話內容時，優先考慮其與當前查詢的相關性
                    - 對於模糊查詢，傾向於檢索以獲得更豐富的上下文
                    - 對於複合查詢（既有通用又有個人化需求），優先選擇檢索

                    **輸出要求：**
                    僅回答以下選項之一：
                    - yes
                    - no
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
            你是一個專業的訊息檢索評估員，專門負責判斷dialogue knowledgebase內的對話內容與使用者查詢的相關性。

            **評估任務：**
            基於使用者查詢的語義意圖和訊息需求，評估dialogue knowledgebase內的對話內容是否能提供有用的訊息來回答或解決該查詢。

            **評估材料：**
            使用者查詢: {query}
            dialogue knowledgebase內的對話內容: {dialogue}

            **相關性判斷框架：**

            **相關性評估維度：**
            1. **主題匹配度** - 對話是否涵蓋查詢的核心主題或概念
            2. **語義相關性** - 對話內容是否在語義上與查詢意圖相關
            3. **訊息實用性** - 對話是否提供能幫助回答查詢的有用訊息
            4. **語境適配性** - 對話情境是否與查詢背景相符

            **判斷標準：**

            **relevant (相關)** - 滿足以下條件之一：
            - 對話直接討論查詢所涉及的主題或概念
            - 對話包含能回答查詢的關鍵訊息或線索
            - 對話提供與查詢語義相關的背景訊息或上下文
            - 對話中的實體、事件或概念與查詢高度相關

            **irrelevant (不相關)** - 符合以下情況：
            - 對話主題與查詢完全無關
            - 對話不包含任何能幫助回答查詢的訊息
            - 對話與查詢在語義和概念層面都缺乏關聯性
            - 對話無法為解決查詢提供任何有用的背景或線索

            **評估步驟：**
            1. 識別查詢的核心概念和訊息需求
            2. 分析對話中的主要主題和關鍵訊息
            3. 評估對話是否包含與查詢語義相關的內容
            4. 判斷對話訊息對回答查詢的實用價值

            **輸出要求：**
            僅回答以下選項之一，不需其他說明：
            - relevant
            - irrelevant
            """
        response = self.generate_content(prompt).lower().strip()
        return IsREL.RELEVANT if "relevant" in response else IsREL.IRRELEVANT
    
    def predict_issup(self, query: str, dialogue: str, candidate_answer: str) -> IsSUP:
        """
        M predicts IsSUP given x, y_t, d
        判斷對話是否支撐候選答案
        """
        prompt = f"""
                    你是一個專業的事實查證員，需要嚴格判斷dialogue knowledgebase內的對話內容是否支撐候選答案中的陳述。

                    **任務說明：**
                    請基於提供的dialogue knowledgebase內的對話內容，評估候選答案的支撐程度。

                    **評估材料：**
                    使用者查詢: {query}
                    候選答案: {candidate_answer}
                    dialogue knowledgebase的對話內容: {dialogue}

                    **判斷標準與定義：**

                    **fully supported (完全支持)** - 必須同時滿足：
                    - 候選答案中的每一個事實陳述都能在對話中找到明確依據
                    - 對話內容與答案陳述完全一致，無矛盾
                    - 所有關鍵訊息都有對話支持

                    **partially supported (部分支撐)** - 符合以下情況：
                    - 候選答案中部分陳述有對話依據，部分缺乏依據
                    - 對話支撐主要論點但缺少細節支撐
                    - 存在輕微不一致但核心訊息正確

                    **no support (無支撐)** - 符合以下情況：
                    - 候選答案的主要陳述在對話中找不到依據
                    - 對話內容與答案陳述存在明顯矛盾
                    - 答案包含對話中未提及的重要訊息

                    **評估步驟：**
                    1. 識別候選答案中的所有關鍵陳述
                    2. 逐一檢查每個陳述在對話中的支持程度
                    3. 評估整體支持一致性
                    4. 根據支持比例和重要性做出最終判斷

                    **輸出要求：**
                    僅回答以下三個選項之一，不需其他解釋：
                    - fully supported
                    - partially supported  
                    - no support
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
        """
        context = f"### 參考對話內容\n{dialogue}" if dialogue else ""
        
        prompt = f"""
                    你是一位嚴格的AI品質評估專家。你的任務是根據一套詳細的負面指標和正面指標，對「候選答案」的有用性進行批判性評分。

                    你的評分必須非常嚴格。大部分的答案可能只會落在2-3分。只有真正完美無瑕的答案才能獲得5分。

                    **評估材料:**
                    ### 使用者查詢
                    {query}

                    ### 候選答案
                    {candidate_answer}

                    {context}

                    ---

                    **思考與評估流程 (Chain-of-Thought):**
                    1.  **檢視負面指標**：首先，檢查答案是否存在任何「扣分項」。
                    2.  **評估正面指標**：接著，根據答案的品質評估其「加分項」。
                    3.  **綜合判斷**：基於上述分析，給出一個最能反映其真實品質的整數分數。

                    ---

                    **評分標準 (詳細定義):**

                    **主要扣分項 (Negative Indicators):**
                    -   **不準確或虛構 (Inaccurate/Hallucinated)**: 包含與事實不符的資訊。
                    -   **答非所問 (Irrelevant)**: 未能直接回應使用者查詢的核心。
                    -   **過於籠統 (Too Generic)**: 提供了常識性、無特定價值的回答。
                    -   **缺乏細節 (Lacks Detail)**: 未能提供足夠的深度或具體資訊。
                    -   **難以理解 (Hard to Understand)**: 語言晦澀、結構混亂。

                    **評分等級定義：**

                    **5分 (卓越):**
                    - **必須同時滿足**:
                        - 完全回答了問題的所有層面。
                        - 提供了具體、可操作且深刻的見解。
                        - 資訊密度高，沒有任何冗言贅字。
                        - **完全沒有**任何上述「扣分項」。

                    **4分 (優秀):**
                    - 回答準確且直接。
                    - 提供了有價值的資訊，但可能在深度或完整性上略有不足。
                    - 可能有極輕微的冗餘。

                    **3分 (合格):**
                    - 基本回答了問題，但較為表面。
                    - 資訊是正確的，但缺乏具體細節。
                    - 是一個安全但價值有限的回答 (A safe but low-value answer)。

                    **2分 (不佳):**
                    - **符合以下任一情況**:
                        - 答案部分偏離主題。
                        - 過於籠統，幾乎沒有提供新資訊。
                        - 雖然相關，但幫助極其有限。

                    **1分 (不可接受):**
                    - **符合以下任一情況**:
                        - 包含明顯的錯誤資訊。
                        - 完全答非所問。
                        - 具有誤導性或產生負面影響。

                    ---

                    **輸出要求:**
                    **只輸出一個整數數字 (1, 2, 3, 4, 或 5)，不要有任何解釋。**

                    你的評分是：
                """
        response = self.generate_content(prompt).strip()
        try:
            score = int(response)
            return IsUSE(score) if 1 <= score <= 5 else IsUSE.MODERATELY_USEFUL
        except:
            return IsUSE.MODERATELY_USEFUL