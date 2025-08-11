
import logging
import os
import time
from typing import Tuple

import torch
import intel_extension_for_pytorch as ipex

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from torch.profiler import profile, ProfilerActivity, record_function


logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger(__name__)

activities = [ProfilerActivity.CPU]
if torch.cuda.is_available():
    device = "cuda"
    activities += [ProfilerActivity.CUDA]
elif torch.xpu.is_available():
    device = "xpu"
    activities += [ProfilerActivity.XPU]
else:
    print(
        "Neither CUDA nor XPU devices are available to demonstrate profiling on acceleration devices"
    )
    import sys

    sys.exit(0)

sort_by_keyword = device + "_time_total"

print(f"device {device}")

def load_model() -> Tuple[AutoTokenizer, AutoModelForCausalLM]:

    # model_name = os.getenv("MODEL_NAME")
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    logger.info(f"Loading model: {model_name}")

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Define the quantization configuration for 8-bit
    # quantization_config = BitsAndBytesConfig(
    #     load_in_8bit=True,
    #     llm_int8_threshold=6.0,
    # )

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",  #! Dynamically balancing between CPU and GPU
        # quantization_config=quantization_config,  #! Quantization
    ).to(device)


    # model.eval()
    # model = torch.quantization.QuantWrapper(model)

    # qconfig = torch.quantization.QConfig(activation=torch.quantization.observer.MinMaxObserver .with_args(qscheme=torch.per_tensor_symmetric),
        # weight=torch.quantization.default_weight_observer)  # weight could also be perchannel


    # model.qconfig = qconfig

    # Prepare model for inserting observer
    # torch.quantization.prepare(model, inplace=True)

    logger.info(f"Model ({model_name}) loaded.")
    return tokenizer, model



def generate_chat_response(
    prompt: str,
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    max_length: int = 3500,
    temperature: float = 0.7,
    top_k: int = 50,
    top_p: float = 0.9,
) -> str:

    # Tokenize the input prompt
    inputs = tokenizer(prompt, return_tensors="pt")

    # Move input tensors to the same device as the model
    device = next(model.parameters()).device
    inputs = {key: value.to(device) for key, value in inputs.items()}

    # Start timing the response generation process
    start_time = time.time()

    # Generate logits and outputs
    with torch.no_grad(), profile(activities=activities, record_shapes=True) as prof, \
        record_function("model_inference"):  # Do not compute grad. descent/no training involved
        # logits = model(**inputs).logits
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            do_sample=True,  # have multi-options (tokens) picks 1 based on prob.
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

        # logger.debug(
            # f"Intermediate logits shape: {logits.shape}"
        # )  # Debugging: inspect logits
    print(prof.key_averages().table(sort_by=sort_by_keyword, row_limit=100))
    prof.export_chrome_trace("trace.json")

    # Calculate the time elapsed for thinking
    elapsed_time = time.time() - start_time
    minutes, seconds = divmod(elapsed_time, 60)
    time_str = f"{int(minutes):02}:{int(seconds):02}"

    # print(f"outputs: {outputs}")
    # Decode the full response
    final_answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Log the thinking time and final response
    logger.info(f"Thinking time: {time_str}")
    logger.info(f"Response from generate_chat_response function:\n{final_answer}")

    return final_answer


def main_chat():
    """Orchestrate the loop"""
    print("Chat with DeepSeek R1! Type 'exit' to end the chat.")

    # Load the model and tokenizer
    tokenizer, model = load_model()

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == "exit":
            break

        # Generate and display the response
        final_output = generate_chat_response(
            prompt=user_input, tokenizer=tokenizer, model=model, max_length=3000
        )
        print(f"DeepSeek (Final Answer): {final_output}")
        logger.info(f"Response: {final_output}")

def main_qa():
    """Orchestrate the loop"""
    print("Chat with DeepSeek R1! Type 'exit' to end the chat.")

    # Load the model and tokenizer
    tokenizer, model = load_model()


    user_input = "how to build up a website in 10 steps:"

    user_input = f"output the answer directly, without the analyse progress: {user_input}"
    # Generate and display the response
    final_output = generate_chat_response(
        prompt=user_input, tokenizer=tokenizer, model=model, max_length=300
    )
    # print(f"DeepSeek (Final Answer): {final_output}")
    # logger.info(f"Response: {final_output}")

if __name__ == "__main__":
    main_qa()