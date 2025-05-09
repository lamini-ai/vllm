from typing import List

import ray
import vllm
from vllm.mome.request import MoMERequest

def do_sample(llm: vllm.LLM, mome_path: str, mome_id: int = 1) -> List[str]:
    prompts = [
        "[user] Write a paragraph on the future of AI. [/user] [assistant]", 
    ]
    sampling_params = vllm.SamplingParams(temperature=0,
                                          max_tokens=256,
                                          stop=["[/assistant]"])
    outputs = llm.generate(
        prompts,
        sampling_params,
        mome_request=MoMERequest(str(mome_id), mome_id, mome_path)
        )
    # Print the outputs.
    generated_texts: List[str] = []

    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        generated_texts.append(generated_text)
        print(f"Prompt: {prompt!r}, \n Generated text: {generated_text!r}")
    return generated_texts

def example_use_mome(base_model_path: str = "meta-llama/Llama-3.1-8B-Instruct",
                     mome_adapter_files: str = "/root/34916/checkpoints/checkpoint-60"):

    llm = vllm.LLM(base_model_path,
                   enable_mome=True,
                   max_num_seqs=16,
                   max_mome_rank=32,
                   max_momes=3,
                   max_model_len=4096,
                   tensor_parallel_size=1,
                   enable_chunked_prefill=True)
    
    res = do_sample(llm, mome_adapter_files, mome_id=1)
    # print(res)

if __name__ == '__main__':
    example_use_mome()