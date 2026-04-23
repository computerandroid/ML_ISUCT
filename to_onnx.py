import torch
from transformers import AutoProcessor, AutoModelForCausalLM

model_id = "kingabzpro/medgemma-brain-cancer"

processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)

model.eval()

dummy_input = torch.randint(0, 100, (1, 256))

torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    input_names=['input_ids'],
    output_names=['logits'],
    dynamic_axes={
        'input_ids': {0: 'batch_size', 1: 'sequence_length'},
        'logits': {0: 'batch_size', 1: 'sequence_length'}
    },
    opset_version=14
)
