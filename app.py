import gradio as gr
from dialogue_rag_chatbot.logging import logger
import main

# 在模組載入時就初始化系統
logger.info("🔄 正在啟動對話分析服務...")
chatbot_interface = main.MainChatbotInterface()

# 創建狀態顯示組件
def get_status():
    return chatbot_interface.get_system_status()

# 創建 Gradio 介面
with gr.Blocks(title="對話分析Chatbot") as demo:
    gr.Markdown("# 🤖 對話分析Chatbot")
    gr.Markdown("基於SAMSum dataset的智慧對話分析系統")

    # 系統狀態顯示
    with gr.Accordion("📊 系統服務狀態", open=False):
        status_display = gr.Textbox(
            value=chatbot_interface.get_system_status(),
            label="系統資訊",
            interactive=False,
            max_lines=10
        )
        refresh_btn = gr.Button("🔄 更新狀態", size="sm")
        refresh_btn.click(fn=get_status, outputs=status_display)

    # 聊天介面
    chatbot = gr.ChatInterface(
        fn=chatbot_interface.chat_response,
        type="messages",
        examples=[
            "在dialogue知識庫中，哪些人談話跟工作有關?",
            "我提供的dialogue知識庫裡，是否有關於戰爭這方面的對話?",
            "找出涉及時間安排的對話，因為他們可能有安排行程的需求"
        ],
        cache_examples=False,
        concurrency_limit=1
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )