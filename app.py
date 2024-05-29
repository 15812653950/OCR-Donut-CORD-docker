from flask import Flask, request, jsonify
from PIL import Image
from transformers import DonutProcessor, VisionEncoderDecoderModel
import traceback
import os
import re
import torch
import uuid
app = Flask(__name__)


model_name = 'jinhybr/OCR-Donut-CORD'
model_path = f'model/{model_name}'
load_flag = 0


@app.route('/',methods=['POST','GET'])
def index():
    return 'Docker container is jinhybr/OCR-Donut-CORD'

##总台链接测试，确保启动正常
@app.route('/connect_test',methods=['POST','GET'])
def connect_test():
    return 'Docker container is up and running!'

@app.route('/load_model',methods=['POST','GET'])
def load_model():
    global processor
    global model
    global load_flag
    global device
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
    load_flag = 1
    return "model is loaded!!!"
    

## 发送结果到总台
##任务执行，发送任务进行处理
@app.route('/predict', methods=['POST','GET'])
def predict():
    
    if load_flag:
        print("model loaded")
    else:
        load_model()
        print("model loading")
    try:
        # Debugging info
        print("Request received")
        print("Headers:", request.headers)
        print("Form data:", request.form)
        print("Files:", request.files)
        result_switch = request.form.get('result_switch', 'on') == 'on'
        # 获取 question 参数
        if 'file' not in request.files:
            return jsonify({"code": 404, "msg": "No file part", "data": {}}), 404


        if 'file' not in request.files:
            return jsonify({"code": 404, "msg": "No file part", "data": {}}), 404

        file = request.files['file']

        # 检查是否有文件被上传
        if file.filename == '':
            return jsonify({"code": 404, "msg": "No selected file", "data": {}}), 404

        # 检查文件类型
        if not (file.filename.endswith('.png') or file.filename.endswith('.jpg')):
            return jsonify({"code": 400, "msg": "Invalid file type. Only .png and .jpg are supported.", "data": {}}), 400

        # Debugging info
        print(f"File received: {file.filename}")
        # 生成 UUID 文件名并保存文件
        name = uuid.uuid4()
        filename = f"{name}.png"
        file.save(filename)

        image = Image.open(filename).convert('RGB')
        
        # 使用处理器准备模型的输入
        pixel_values = processor(image, return_tensors="pt").pixel_values

        # 准备解码器的输入
        task_prompt = "<s_cord-v2>"
        decoder_input_ids = processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids

        # 将输入移动到适当的设备上
        pixel_values = pixel_values.to(device)
        decoder_input_ids = decoder_input_ids.to(device)

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
        if result_switch is None or result_switch:
            sequence = re.sub(r'<.*?>', '', sequence)  # 删除所有的格式标签
        print(sequence)

        os.remove(filename)
        return jsonify({"code": 200, "msg": "Success", "data": sequence}), 200

    except Exception as e:
        traceback.print_exc()
        return jsonify({"code": 404, "msg": str(e), "data": {}}), 404

# Run the app if this file is executed
if __name__ == "__main__":
    app.run(port = 6791, host='0.0.0.0', debug=True)

