import gradio as gr
from openai import OpenAI
from image_utils import IMAGE
from rag_utils import RAG
from prompt_utils import create_rag_prompt, create_image_prompt, create_default_prompt

class GRADIO:
    def __init__(self, api_key):
        self.rag = None
        self.image = None
        self.pdf_path = None
        self.image_path = None
        self.client = OpenAI(
            api_key=api_key,
            base_url="http://axonflow.xyz/v1",
            timeout=30.0,
            max_retries=3
        )

        with gr.Blocks(fill_height=True) as self.demo:
            chatbot = gr.Chatbot(elem_id="chatbot", bubble_full_width=False, type="messages", scale=1)
            chat_input = gr.MultimodalTextbox(
                interactive=True,
                file_count="multiple",
                file_types=[".pdf", ".jpg", ".jpeg", ".png"],
                placeholder="Enter message or upload file...",
                show_label=False
            )
            history_state = gr.State([])  # 사용자별 채팅 기록 상태 저장

            chat_msg = chat_input.submit(
                self.add_message,
                [chat_input, chatbot, history_state],
                [chatbot, chat_input, history_state]
            )
            bot_msg = chat_msg.then(
                self.bot_response,
                [chatbot, history_state],
                [chatbot]
            )
            bot_msg.then(lambda: gr.MultimodalTextbox(interactive=True), None, [chat_input])

            with gr.Row():
                clear = gr.ClearButton([chat_input, chatbot, history_state])

    def add_message(self, message, history, state):
        if message["files"]:
            for file_path in message["files"]:
                if file_path.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.upload_file_image(file_path)
                    self.image_path = file_path
                elif file_path.lower().endswith('.pdf'):
                    self.upload_file_rag(file_path)
                    self.pdf_path = file_path
                state.append({"role": "user", "content": {"path": file_path}})

        if message["text"] is not None:
            state.append({"role": "user", "content": message["text"]})
        return state, gr.MultimodalTextbox(value=None, interactive=False), state

    def bot_response(self, history, state):
        if not state:
            return history

        last_message = next((msg for msg in reversed(state)
                             if msg["role"] == "user"), None)

        if last_message:
            if isinstance(last_message["content"], dict) and "path" in last_message["content"]:
                return history

            _, updated_state = self.respond(last_message["content"], state)
            return updated_state
        return history

    def upload_file_rag(self, file_path):
        print(f"Loading PDF: {file_path}")
        self.rag = RAG(file_path)

    def upload_file_image(self, file_path):
        print(f"Loading Image: {file_path}")
        if file_path is None:
            return
        self.image = IMAGE(file_path)

    def respond(self, message, chat_history):
        if self.rag is not None:
            results = self.rag.search(message)
            prompt = create_rag_prompt(message, self.pdf_path, results)
        elif self.image is not None:
            encoded_image = self.image.encode_base64_content_from_file()
            prompt = create_image_prompt(message, encoded_image)
        else:
            prompt = create_default_prompt(message)

        # prompt = chat_history[:-1]+prompt

        print("Prompt:")
        for w in prompt:
            print('\t' + str(w))

        try:
            chat_completion = self.client.chat.completions.create(
                messages=prompt,
                model="Qwen/Qwen2-VL-2B-Instruct",
                stream=False,
            )
            res = chat_completion.choices[0].message.content
        except Exception as e:
            error_msg = f"API 요청 중 오류 발생: {str(e)}"
            print(error_msg)
            if "Connection error" in str(e):
                res = "서버 연결에 실패했습니다. 잠시 후 다시 시도해주세요."
            elif "timeout" in str(e).lower():
                res = "요청 시간이 초과되었습니다. 잠시 후 다시 시도해주세요."
            else:
                res = "죄송합니다. 요청을 처리하는 중 오류가 발생했습니다."

        print("Response:", res, '\n\n\n')
        chat_history.append({"role": "assistant", "content": res})
        return "", chat_history

if __name__ == "__main__":
    g = GRADIO(api_key="EMPTY").demo.launch(debug=True)