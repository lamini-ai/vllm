from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors.torch import load_file
from collections import defaultdict

def print_base_model_structure(model_path=""):
    """
    Print the structure of the base model.
    """
    if not model_path:
        model_path = "/root/.cache/huggingface/hub/models--meta-llama--Llama-3.2-3B-Instruct/snapshots/0cb88a4f764b7a12671c53f0838cd831a0843b95/"
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="auto"
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    print(model)
    print(model.config)

    for module_name, module in model.named_modules():
        print("module_name: ", module_name)
        print(module)


def print_base_mome_adapter_structure(adapter_path=""):
    if not adapter_path:
        adapter_path = "/app/lamini/jobs/34916/checkpoints/checkpoint-60/adapter_model.safetensors"

    state_dict = load_file(adapter_path)
    structure = defaultdict(lambda: defaultdict(list))

    for k, v in state_dict.items():
        parts = k.split(".")
        if "layers" in parts:
            layer_idx = parts[parts.index("layers") + 1]
            module = parts[parts.index("layers") + 2]
            param = ".".join(parts[parts.index("layers") + 3:])
            structure[layer_idx][module].append((param, v.shape))
        else:
            structure["others"][parts[-2]].append((".".join(parts[-2:]), v.shape))

    for layer, modules in sorted(structure.items(), key=lambda x: (x[0] != "others", int(x[0]) if x[0] != "others" else -1)):
        print(f"\n====== Layer {layer} ======")
        for module, params in modules.items():
            print(f"  [{module}]")
            for name, shape in params:
                print(f"    - {name} : {shape}")


if __name__ == "__main__":
    print_base_model_structure()
    print_base_mome_adapter_structure()
