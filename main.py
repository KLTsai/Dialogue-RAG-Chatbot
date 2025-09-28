from legal_rag_chatbot.logging import logger
from legal_rag_chatbot.config.configuration import ConfigurationManager
from legal_rag_chatbot.components.rag_sys import DialogueRAGSystem
import os

def main():
    try:
        config = ConfigurationManager()
        rag_system = DialogueRAGSystem(config)
        rag_system.build_knowledge_base()
        
        test_query = "哪些人談話跟工作有關?"
        result = rag_system.query(test_query)

        print(f"🤖 回答: {result['answer']}")
        print(f"📊 檢索決策: {result['retrieve_decision']}")
        print(f"📁 參考來源: {result.get('sources', [])}")
        print(f"⏱️ 處理時間: {result['processing_time']:.2f}秒")

        if 'support_level' in result:
            print(f"🎯 支撐程度: {result['support_level']}")
        if 'usefulness_score' in result:
            print(f"⭐ 有用性評分: {result['usefulness_score']}/5")
        if 'reference_dialogue' in result:
            print(f"💬 參考對話: {result['reference_dialogue']}")

        # 顯示系統統計
        print(f"\n{'='*60}")
        print("📈 系統統計資訊")
        print(f"{'='*60}")
        stats = rag_system.get_system_stats()
        for key, value in stats.items():
            print(f"{key}: {value}")

    except Exception as e:
        raise e

if __name__ == "__main__":
    
    main()