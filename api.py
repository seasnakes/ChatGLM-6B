from fastapi import FastAPI, Request
from transformers import AutoTokenizer, AutoModel
import uvicorn, json, datetime
import torch
# 指定显卡 
DEVICE = "cuda"
DEVICE_ID = "0"
CUDA_DEVICE = f"{DEVICE}:{DEVICE_ID}" if DEVICE_ID else DEVICE

# 检查显卡cuda是否可用
# 清空cuda占用显存
def torch_gc():
    if torch.cuda.is_available():
        with torch.cuda.device(CUDA_DEVICE):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

# 创建FastAPI实例
app = FastAPI()


@app.post("/")
async def create_item(request: Request):
    global model, tokenizer
    # 异步接收请求将请求json序列化
    json_post_raw = await request.json()
    json_post = json.dumps(json_post_raw)
    json_post_list = json.loads(json_post)
    # 从post来的json中读取prompt字段
    prompt = json_post_list.get('prompt')
     # 从post来的json中读取history（历史记录）字段
    history = json_post_list.get('history')
    max_length = json_post_list.get('max_length')
    top_p = json_post_list.get('top_p')
    temperature = json_post_list.get('temperature')
    #定义http 返回体 内容为模型推理完后的结果
    response, history = model.chat(tokenizer,
                                   prompt,
                                   history=history,
                                   max_length=max_length if max_length else 2048,
                                   top_p=top_p if top_p else 0.7,
                                   temperature=temperature if temperature else 0.95)
    #时间戳
    now = datetime.datetime.now()
    time = now.strftime("%Y-%m-%d %H:%M:%S")
    answer = {
        "response": response,
        "history": history,
        "status": 200,
        "time": time
    }
    log = "[" + time + "] " + '", prompt:"' + prompt + '", response:"' + repr(response) + '"'
    print(log)
    #pytorch的垃圾回收
    torch_gc()
    return answer


if __name__ == '__main__':
    # 从预训练权重中加载 tokenizer 和 model
    tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
    model = AutoModel.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True).half().cuda()
    model.eval()
    # 用uvicorn （python实习的一款http服务器）部署在本机指定端口上 示例里为8000端口
    uvicorn.run(app, host='0.0.0.0', port=8000, workers=1)
