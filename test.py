from PIL import Image
from transformers import DonutProcessor, VisionEncoderDecoderModel
import os
import re
import torch
model_name = 'jinhybr/OCR-Donut-CORD'
model_path = f'model/{model_name}'
if os.path.exists(model_path):
    model = VisionEncoderDecoderModel.from_pretrained(model_path)
    processor = DonutProcessor.from_pretrained(model_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
else:
    model = VisionEncoderDecoderModel.from_pretrained(model_name)
    processor = DonutProcessor.from_pretrained(model_name)
    model.save_pretrained(model_path)
    processor.save_pretrained(model_path)
    model = VisionEncoderDecoderModel.from_pretrained(model_path)
    processor = DonutProcessor.from_pretrained(model_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
image_path = "1.png"
image = Image.open(image_path).convert('RGB')

# 使用处理器准备模型的输入
pixel_values = processor(image, return_tensors="pt").pixel_values

# 准备解码器的输入
task_prompt = "<s_cord-v2>"
decoder_input_ids = processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids

# 将输入移动到适当的设备上
pixel_values = pixel_values.to(device)
decoder_input_ids = decoder_input_ids.to(device)

# 使用模型进行推理
# 使用模型进行推理
outputs = model.generate(
    pixel_values,
    decoder_input_ids=decoder_input_ids,
    max_length=model.decoder.config.max_position_embeddings,
    pad_token_id=processor.tokenizer.pad_token_id,
    eos_token_id=processor.tokenizer.eos_token_id,
    use_cache=True,
    bad_words_ids=[[processor.tokenizer.unk_token_id]],
    return_dict_in_generate=True,
)

sequence = processor.batch_decode(outputs.sequences)[0]
sequence = re.sub(r'<.*?>', '', sequence)  # 删除所有的格式标签
print(sequence)
# print(sequence) result
# <s_cord-v2><s_menu><s_nm> Section D BBQ : (B)</s_nm><s_unitprice> Business Nature (may choose more than one item)</s_nm><s_unitprice> @FGRAMBI : CREAMENTRAL CREAMEDICAL CHOCOCOLORANET Utilities or Gothers (Blee sechy) Oma Goreng</s_nm></s_sub></s_menu><s_total><s_total_price> Orian</s_nm><s_cnt> 1</s_cnt><s_price> Ozona</s_nm></s_total></s>
# sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
# sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()  # remove first task start token

print(sequence)
# 解码模型的输出
# sequence = processor.batch_decode(outputs.sequences)[0]
# sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
# sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()  # remove first task start token

# # 将输出转换为 JSON
# res = processor.token2json(sequence)

# print(res)