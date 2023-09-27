from typing import List
from fastchat.conversation import (
    Conversation, 
    SeparatorStyle, 
    get_conv_template, 
    register_conv_template,
)
from fastchat.model.model_adapter import (
    BaseModelAdapter,
    model_adapters, 
    get_conversation_template, 
    register_model_adapter,
)

class TKAdapter(BaseModelAdapter):
    def match(self, model_path: str) -> bool:
        return "tk" in model_path
    
    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("tk")
    
class OpenAIAdapter(BaseModelAdapter):
    def match(self, model_path: str):
        return any(model_path.startswith(s) for s in ("gpt-3.5-turbo", "gpt-4",))

    def load_model(self, model_path: str, from_pretrained_kwargs: dict):
        raise NotImplementedError()

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("zero_shot")


custom_adapters: List[BaseModelAdapter] = [
    TKAdapter,
    OpenAIAdapter,
]
custom_conversations: List[Conversation] = [
    Conversation(
        name="tk",
        system_message="Definition: write an output that appropriately completes the following request. ",
        roles=("Input", "Output"),
        messages=(),
        offset=2,
        sep_style=SeparatorStyle.ADD_COLON_SINGLE,
        sep="\n",
        stop_str=["User"],
    ),
]

def register_custom_adapters():
    base = model_adapters.pop()
    for m in custom_adapters:
        register_model_adapter(m)
    model_adapters.append(base)

def register_custom_conversations():
    for c in custom_conversations:
        register_conv_template(c)
