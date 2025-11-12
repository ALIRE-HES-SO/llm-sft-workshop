# ----------------------------------------------------------------------------------------------------
import gradio as gr
# ----------------------------------------------------------------------------------------------------
from openai import OpenAI
# ----------------------------------------------------------------------------------------------------
class ChatManager:

    def __init__(
        self,
        api_key,
        api_base_url,
        system_prompt
    ):
        self.STREAM = False
        self.client = OpenAI(
            api_key=api_key,
            base_url=api_base_url
        )
        self.system_prompt = system_prompt

    def chat(
        self,
        messages
    ):
        self.STREAM = True

        completion = self.client.chat.completions.create(
            model="local",
            stream=True,
            messages=messages
        )

        assistant_response = gr.ChatMessage(
            role="assistant",
            content=""
        )
        messages.append(assistant_response)

        for chunk in completion:

            if not self.STREAM:
                break

            delta = chunk.choices[0].delta
            updated = False

            if getattr(delta, "content", None):
                assistant_response.content += delta.content
                updated = True

            if updated:
                yield messages

    def submit(
        self,
        message,
        history
    ):
        if len(history) == 0:
            history.append(
                gr.ChatMessage(
                    role="system",
                    content=self.system_prompt
                )
            )
        
        history.append(
            gr.ChatMessage(
                role="user",
                content=message
            )
        )
        return self.disable_chat_input(), history

    def disable_chat_input(
        self
    ):
        return gr.update(
            value="",
            stop_btn=True,
            submit_btn=False,
            interactive=False
        )

    def enable_chat_input(
        self
    ):
        return gr.update(
            value="",
            stop_btn=False,
            submit_btn=True,
            interactive=True
        )

    def stop_chat(
        self
    ):
        self.STREAM = False

    def handle_edit(
        self,
        history,
        edit_data: gr.EditData
    ):
        new_history = None
        if isinstance(edit_data.index, int):
            new_history = history[:edit_data.index + 1]
            new_history[-1]['content'] = edit_data.value
        return self.disable_chat_input(), new_history

    def handle_retry(
        self,
        history,
        retry_data: gr.RetryData
    ):
        new_history = None
        if isinstance(retry_data.index, int):
            new_history = history[:retry_data.index + 1]
        return self.disable_chat_input(), new_history
# ----------------------------------------------------------------------------------------------------