from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",                                                                                              bnb_4bit_compute_dtype="float16",
    bnb_4bit_use_double_quant=True,                                                                                     ) 
llm = HuggingFacePipeline.from_model_id(
    model_id="upstage/SOLAR-10.7B-Instruct-v1.0",
    task="text-generation",
    pipeline_kwargs=dict(
        max_new_tokens=512,
        do_sample=False,
        repetition_penalty=1.03,
        return_full_text=False,
    ),
    model_kwargs={"quantization_config": quantization_config},
)

chat_model = ChatHuggingFace(llm=llm)

ai_message = chat_model.invoke("what is huggingface?")
