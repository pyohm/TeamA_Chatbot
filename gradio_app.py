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
        self.client = OpenAI(api_key=api_key, base_url="http://axonflow.xyz/v1")

        with gr.Blocks(fill_height=True) as self.demo:
            chatbot = gr.Chatbot(elem_id="chatbot", bubble_full_width=False, type="messages", scale=1)
            chat_input = gr.MultimodalTextbox(
                interactive=True, 
                file_count="multiple", 
                file_types=[".pdf", ".jpg", ".jpeg", ".png"], 
                placeholder="Enter message or upload file...", 
                show_label=False
            )
            
            chat_msg = chat_input.submit(
                self.add_message, 
                [chat_input, chatbot], 
                [chatbot, chat_input]
            )
            bot_msg = chat_msg.then(
                self.bot_response,
                chatbot,
                chatbot
            )
            bot_msg.then(lambda: gr.MultimodalTextbox(interactive=True), None, [chat_input])

            with gr.Row():
                clear = gr.ClearButton([chat_input, chatbot])

    def add_message(self, message, history):
        if message["files"]:
            for file_path in message["files"]:
                if file_path.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.upload_file_image(file_path)
                    self.image_path = file_path
                elif file_path.lower().endswith('.pdf'):
                    self.upload_file_rag(file_path)
                    self.pdf_path = file_path
                history.append({"role": "user", "content": {"path": file_path}})
        
        if message["text"] is not None:
            history.append({"role": "user", "content": message["text"]})
        return history, gr.MultimodalTextbox(value=None, interactive=False)

    def bot_response(self, history):
        if not history:
            return history
        
        last_message = next((msg for msg in reversed(history) 
                            if msg["role"] == "user"), None)
        
        if last_message:
            if isinstance(last_message["content"], dict) and "path" in last_message["content"]:
                return history
            
            _, updated_history = self.respond(last_message["content"], history)
            return updated_history
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

        print("Prompt:")
        for w in prompt:
            print('\t' + str(w))
            
        chat_completion = self.client.chat.completions.create(
            messages=prompt,
            model="Qwen/Qwen2-VL-2B-Instruct",
            stream=False,
        )

        res = chat_completion.choices[0].message.content
        print("Response:", res, '\n\n\n')

        chat_history.append({"role": "assistant", "content": res})
        return "", chat_history

if __name__ == "__main__":
    g = GRADIO(api_key="EMPTY").demo.launch(debug=True)