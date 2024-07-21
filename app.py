import gradio as gr
from huggingface_hub import InferenceClient

client = InferenceClient("HuggingFaceH4/zephyr-7b-beta")

def respond(
    message,
    history,
    system_message,
    max_tokens,
    temperature,
    top_p,
):
    system_message = "Welcome to the Tech Innovation Consultant! I'm here to provide insights and guidance on technology innovations, digital transformation strategies, future trends, and more. Whether you're looking for advice on emerging technologies, innovation frameworks, or business strategy in the digital age, feel free to ask. How can I assist you today?"
    messages = [{"role": "system", "content": system_message}]

    for val in history:
        if val[0]:
            messages.append({"role": "user", "content": val[0]})
        if val[1]:
            messages.append({"role": "assistant", "content": val[1]})

    messages.append({"role": "user", "content": message})

    response = ""

    for message in client.chat_completion(
        messages,
        max_tokens=max_tokens,
        stream=True,
        temperature=temperature,
        top_p=top_p,
    ):
        token = message.choices[0].delta.content

        response += token
        yield response

demo = gr.ChatInterface(
    respond,
    additional_inputs=[
        gr.Textbox(value="Welcome to the Tech Innovation Consultant!", label="System message"),
        gr.Slider(minimum=1, maximum=2048, value=512, step=1, label="Max new tokens"),
        gr.Slider(minimum=0.1, maximum=4.0, value=0.7, step=0.1, label="Temperature"),
        gr.Slider(
            minimum=0.1,
            maximum=1.0,
            value=0.95,
            step=0.05,
            label="Top-p (nucleus sampling)",
        ),
    ],
    examples=[
        ["What are the latest trends in artificial intelligence?"],
        ["Can you explain blockchain technology?"],
        ["How can businesses leverage IoT for growth?"],
        ["What are some innovative uses of machine learning in healthcare?"],
        ["How can startups implement a successful digital transformation strategy?"],
        ["What are the key components of a technology innovation roadmap?"],
        ["How does 5G technology impact the future of telecommunications?"],
        ["Can you recommend strategies for cybersecurity in IoT devices?"],
        ["How can AI optimize supply chain management?"],
        ["What are the advantages of cloud computing for businesses?"],
        ["How can companies innovate through data analytics?"],
    ],
    title='Tech Innovation Consultant'
)

if __name__ == "__main__":
    demo.launch()
