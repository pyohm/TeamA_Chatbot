import gradio as gr
import openai
from  openai import OpenAI
import base64

# OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://axonflow.xyz/v1"

client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key=openai_api_key,
    base_url=openai_api_base,
)


def main():

    models = client.models.list()
    model = models.data[0].id
    print(f"Using model: {model}")
    
    
    def encode_base64_content_from_file(image_path: str) -> str:
        """Encode a content retrieved from image_path to base64 format."""
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        return encoded_string

    # Image URL and text input inference
    def run_image_and_text_inference(image_base64, question) -> str:
        # Constructing the messages with both text and image content
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {"type": "image_url", 
                     "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
                ],
            }
        ]

        # Send the combined input to the model
        chat_completion = client.chat.completions.create(
            model=model,  # Adjust model as per availability
            messages=messages,
            max_tokens=128,  # Configurable for more detailed responses
        )

        result = chat_completion.choices[0].message.content
        return result

    def generate_response(image_path, question):
        # The 'image' variable is a file path string, so use it directly
        image_base64 = encode_base64_content_from_file(image_path)
        response = run_image_and_text_inference(image_base64, question)
        return response


    with gr.Blocks() as demo:
        gr.Markdown("# Image URL and Text Input Demo")
        
        # Image and text input components
        image_input = gr.Image(label="Upload an Image", type="filepath")
        text_input = gr.Textbox(label="Ask a Question", placeholder="Enter your question about the image")
        
        # Output component
        output_box = gr.Textbox(label="Response", placeholder="Generated response from the model")

        # Trigger response generation when both inputs are provided
        submit_btn = gr.Button("Submit")
        submit_btn.click(generate_response, inputs=[image_input, text_input], outputs=[output_box])

    demo.queue().launch(share=False)

if __name__ == "__main__":
    main()
