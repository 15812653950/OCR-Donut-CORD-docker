import requests
import json

# 服务器地址
url = "http://localhost:6791"

# 发送预测请求
# 假设我们有一个名为'test.png'的图片文件
file_path = "test.png"

# 打开文件并发送请求
with open(file_path, "rb") as f:
    files = {"file": f}
    # "result_switch" 参数决定是否清理格式标签，'on' 表示清理，'off' 表示不清理
    #可以不传，不传我就默认清理
    data = {"result_switch": 'on'}
    response = requests.post(f"{url}/predict", files=files, data=data)
    # 获取响应中的JSON数据
    json_data = response.json()
    # 打印"data"键的值，这是预测的结果
    print(json_data["data"])
    # 将请求的数据和响应一起保存到txt文件中
    with open("output.txt", "w") as out_file:
        out_file.write("Request Data:\n")
        out_file.write(json.dumps(data, indent=4))
        out_file.write("\n\nResponse:\n")
        out_file.write(json.dumps(response.json(), indent=4))