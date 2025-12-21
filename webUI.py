# 导入 Gradio 库
import gradio as gr
# 导入 requests 库
import requests
# 导入 json 库
import json
# 导入 logging 库
import logging
# 导入 re 库
import re
# 导入 uuid 库
import uuid
from datetime import datetime

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

url = "http://127.0.0.1:8000/v1/chat/completions"
headers = {"Content-Type": "application/json"}
stream_flag = False 

users_db = {}
user_id_map = {}

def generate_unique_user_id(username):
    if username not in user_id_map:
        user_id = str(uuid.uuid4())
        while user_id in user_id_map.values():
            user_id = str(uuid.uuid4())
        user_id_map[username] = user_id
    return user_id_map[username]

def generate_unique_conversation_id(username):
    return f"{username}_{uuid.uuid4()}"

# ==========================================
# 核心修改：完全适配 Gradio 3.50.2 的列表格式
# ==========================================
def send_message(user_message, history, user_id, conversation_id, username):
    data = {
        "messages": [{"role": "user", "content": user_message}],
        "stream": stream_flag,
        "userId": user_id,
        "conversationId": conversation_id
    }

    # 【重点修正】这里必须用列表嵌套 [[user, ai]]，不能用字典！
    # 如果 history 是 None，先初始化为空列表
    if history is None:
        history = []
        
    history = history + [[user_message, "正在生成回复..."]]
    yield history, history, None

    if username and conversation_id:
        if not users_db[username]["conversations"][conversation_id].get("title_set", False):
            new_title = user_message[:20] if len(user_message) > 20 else user_message
            users_db[username]["conversations"][conversation_id]["title"] = new_title
            users_db[username]["conversations"][conversation_id]["title_set"] = True

    def format_response(full_text):
        formatted_text = re.sub(r'<think>', '**思考过程**：\n', full_text)
        formatted_text = re.sub(r'</think>', '\n\n**最终回复**：\n', full_text)
        return formatted_text.strip()

    if stream_flag:
        assistant_response = ""
        try:
            with requests.post(url, headers=headers, data=json.dumps(data), stream=True) as response:
                for line in response.iter_lines():
                    if line:
                        json_str = line.decode('utf-8').strip("data: ")
                        if not json_str: continue
                        if json_str.startswith('{') and json_str.endswith('}'):
                            try:
                                response_data = json.loads(json_str)
                                if 'delta' in response_data['choices'][0]:
                                    content = response_data['choices'][0]['delta'].get('content', '')
                                    formatted_content = format_response(content)
                                    assistant_response += formatted_content
                                    # 更新最后一条记录的 AI 回复部分
                                    history[-1][1] = assistant_response
                                    yield history, history, None
                                if response_data.get('choices', [{}])[0].get('finish_reason') == "stop":
                                    break
                            except json.JSONDecodeError:
                                history[-1][1] = "解析错误"
                                yield history, history, None
                                break
        except Exception:
            history[-1][1] = "请求失败"
            yield history, history, None

    else:
        try:
            response = requests.post(url, headers=headers, data=json.dumps(data))
            response_json = response.json()
            assistant_content = response_json['choices'][0]['message']['content']
            formatted_content = format_response(assistant_content)
            
            # 【重点修正】更新最后一条记录的 AI 回复部分
            history[-1][1] = formatted_content
            yield history, history, None
        except Exception as e:
            history[-1][1] = f"错误: {str(e)}"
            yield history, history, None

# 以下辅助函数保持不变
def register(username, password):
    try:
        r = requests.post(
            "http://127.0.0.1:8000/auth/register",
            json={"username": username, "password": password},
            timeout=5
        )

        # 成功
        if r.status_code == 200:
            return "注册成功，请登录"

        # 非 JSON 响应（HTML / 空）
        if not r.headers.get("content-type", "").startswith("application/json"):
            return f"注册失败：{r.text or '后端无响应'}"

        # JSON 错误信息
        return r.json().get("detail", "注册失败")

    except requests.exceptions.ConnectionError:
        return "无法连接后端服务（8000）"
    except Exception as e:
        return f"注册异常：{str(e)}"



def login(username, password):
    try:
        r = requests.post(
            "http://127.0.0.1:8000/auth/login",
            json={"username": username, "password": password},
            timeout=5
        )

        if r.status_code != 200:
            if r.headers.get("content-type", "").startswith("application/json"):
                return False, None, None, None, r.json().get("detail", "登录失败")
            return False, None, None, None, "登录失败"

        data = r.json()
        user_id = data["user_id"]

        conversation_id = generate_unique_conversation_id(username)
        create_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if username not in users_db:
            users_db[username] = {"conversations": {}}

        users_db[username]["conversations"][conversation_id] = {
            "history": [],
            "title": "创建新的聊天",
            "create_time": create_time,
            "title_set": False
        }

        return True, username, user_id, conversation_id, "登录成功"

    except requests.exceptions.ConnectionError:
        return False, None, None, None, "无法连接后端服务（8000）"



def new_conversation(username):
    if username not in users_db: return "请先登录！", None
    conversation_id = generate_unique_conversation_id(username)
    create_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    users_db[username]["conversations"][conversation_id] = {
        "history": [], "title": "创建新的聊天", "create_time": create_time, "title_set": False
    }
    return "新会话创建成功！", conversation_id

def get_conversation_list(username):
    if username not in users_db or not users_db[username]["conversations"]: return ["请选择历史会话"]
    conv_list = []
    for conv_id, details in users_db[username]["conversations"].items():
        title = details.get("title", "未命名会话")
        create_time = details.get("create_time", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        conv_list.append(f"{title} - {create_time}")
    return ["请选择历史会话"] + conv_list

def extract_conversation_id(selected_option, username):
    if selected_option == "请选择历史会话" or not username in users_db: return None
    for conv_id, details in users_db[username]["conversations"].items():
        title = details.get("title", "未命名会话")
        create_time = details.get("create_time", "")
        if f"{title} - {create_time}" == selected_option: return conv_id
    return None

def load_conversation(username, selected_option):
    if selected_option == "请选择历史会话" or not username in users_db: return []
    conversation_id = extract_conversation_id(selected_option, username)
    if conversation_id in users_db[username]["conversations"]:
        return users_db[username]["conversations"][conversation_id]["history"]
    return []

# ==========================================
# Gradio 3.50.2 界面定义
# ==========================================
with gr.Blocks(title="聊天助手", css="""
    .login-container { max-width: 400px; margin: 0 auto; padding-top: 100px; }
    .modal { position: fixed; top: 20%; left: 50%; transform: translateX(-50%); background: white; padding: 20px; max-width: 400px; box-shadow: 0 4px 8px rgba(0,0,0,0.2); border-radius: 8px; z-index: 1000; }
    .chat-area { padding: 20px; height: 80vh; }
    .header { display: flex; justify-content: space-between; align-items: center; padding: 10px; }
    .header-btn { margin-left: 10px; padding: 5px 10px; font-size: 14px; }
""") as demo:
    
    # 状态定义
    logged_in = gr.State(False)
    current_user = gr.State(None)
    current_user_id = gr.State(None)
    current_conversation = gr.State(None)
    chatbot_history = gr.State([])
    conversation_title = gr.State("创建新的聊天")

    # 登录页
    with gr.Column(visible=True, elem_classes="login-container") as login_page:
        gr.Markdown("## 聊天助手")
        login_username = gr.Textbox(label="用户名", placeholder="请输入用户名")
        login_password = gr.Textbox(label="密码", placeholder="请输入密码", type="password")
        with gr.Row():
            login_button = gr.Button("登录", variant="primary")
            register_button = gr.Button("注册", variant="secondary")
        login_output = gr.Textbox(label="结果", interactive=False)

    # 聊天页
    with gr.Column(visible=False) as chat_page:
        with gr.Row(elem_classes="header"):
            welcome_text = gr.Markdown("### 欢迎，")
            with gr.Row():
                new_conv_button = gr.Button("新建会话", elem_classes="header-btn", variant="secondary")
                history_button = gr.Button("历史会话", elem_classes="header-btn", variant="secondary")
                logout_button = gr.Button("退出登录", elem_classes="header-btn", variant="secondary")

        with gr.Column(elem_classes="chat-area"):
            title_display = gr.Markdown("## 会话标题", elem_id="title-display")
            chatbot = gr.Chatbot(label="聊天对话", height=450)
            with gr.Row():
                message = gr.Textbox(label="消息", placeholder="输入消息并按 Enter 发送", scale=8, container=False)
                send = gr.Button("发送", scale=2)

    # 弹窗定义
    with gr.Column(visible=False, elem_classes="modal") as register_modal:
        reg_username = gr.Textbox(label="用户名", placeholder="请输入用户名")
        reg_password = gr.Textbox(label="密码", placeholder="请输入密码", type="password")
        with gr.Row():
            reg_button = gr.Button("提交注册", variant="primary")
            close_button = gr.Button("关闭", variant="secondary")
        reg_output = gr.Textbox(label="结果", interactive=False)

    with gr.Column(visible=False, elem_classes="modal") as history_modal:
        gr.Markdown("### 会话历史")
        conv_dropdown = gr.Dropdown(label="选择历史会话", choices=["请选择历史会话"], value="请选择历史会话")
        load_conv_button = gr.Button("加载会话", variant="primary")
        close_history_button = gr.Button("关闭", variant="secondary")

    # 辅助函数
    def show_register_modal(): return gr.update(visible=True)
    def hide_register_modal(): return gr.update(visible=False)
    def show_history_modal(username): return gr.update(visible=True), gr.update(choices=get_conversation_list(username), value="请选择历史会话")
    def hide_history_modal(): return gr.update(visible=False)
    def logout(): return False, None, None, gr.update(visible=True), gr.update(visible=False), "已退出登录", [], None, [], "创建新的聊天"
    def update_welcome_text(username): return gr.update(value=f"### 欢迎，{username}")
    def update_title_display(title): return gr.update(value=f"## {title}")

    # 事件绑定
    register_button.click(show_register_modal, None, register_modal)
    close_button.click(hide_register_modal, None, register_modal)
    reg_button.click(register, [reg_username, reg_password], reg_output)

    login_button.click(login, [login_username, login_password], [logged_in, current_user, current_user_id, current_conversation, login_output]).then(lambda logged: (gr.update(visible=not logged), gr.update(visible=logged)), [logged_in], [login_page, chat_page]).then(update_welcome_text, [current_user], welcome_text).then(lambda username, conv_id: users_db[username]["conversations"][conv_id]["history"] if username and conv_id else [], [current_user, current_conversation], chatbot_history).then(lambda username, conv_id: users_db[username]["conversations"][conv_id].get("title", "创建新的聊天") if username and conv_id else "创建新的聊天", [current_user, current_conversation], conversation_title).then(update_title_display, [conversation_title], title_display)
    logout_button.click(logout, None, [logged_in, current_user, current_user_id, login_page, chat_page, login_output, chatbot, current_conversation, chatbot_history, conversation_title])
    history_button.click(show_history_modal, [current_user], [history_modal, conv_dropdown])
    close_history_button.click(hide_history_modal, None, history_modal)
    new_conv_button.click(new_conversation, [current_user], [login_output, current_conversation]).then(lambda: [], None, chatbot).then(lambda: [], None, chatbot_history).then(lambda: "创建新的聊天", None, conversation_title).then(update_title_display, [conversation_title], title_display)
    load_conv_button.click(load_conversation, [current_user, conv_dropdown], chatbot).then(lambda user, conv: extract_conversation_id(conv, user), [current_user, conv_dropdown], current_conversation).then(lambda username, conv: users_db[username]["conversations"][extract_conversation_id(conv, username)].get("title", "创建新的聊天") if username and conv else "创建新的聊天", [current_user, conv_dropdown], conversation_title).then(update_title_display, [conversation_title], title_display).then(hide_history_modal, None, history_modal)

    def update_history(chatbot_output, history, user, conv_id):
        if user and conv_id: users_db[user]["conversations"][conv_id]["history"] = chatbot_output
        return chatbot_output

    send.click(send_message, [message, chatbot_history, current_user_id, current_conversation, current_user], [chatbot, chatbot_history, conversation_title]).then(update_history, [chatbot, chatbot_history, current_user, current_conversation], chatbot_history).then(lambda username, conv_id: users_db[username]["conversations"][conv_id].get("title", "创建新的聊天") if username and conv_id else "创建新的聊天", [current_user, current_conversation], conversation_title).then(update_title_display, [conversation_title], title_display).then(lambda: "", None, message)
    message.submit(send_message, [message, chatbot_history, current_user_id, current_conversation, current_user], [chatbot, chatbot_history, conversation_title]).then(update_history, [chatbot, chatbot_history, current_user, current_conversation], chatbot_history).then(lambda username, conv_id: users_db[username]["conversations"][conv_id].get("title", "创建新的聊天") if username and conv_id else "创建新的聊天", [current_user, current_conversation], conversation_title).then(update_title_display, [conversation_title], title_display).then(lambda: "", None, message)

if __name__ == "__main__":
    # 【重点修正】加上 .queue() 解决 ValueError
    demo.queue().launch(server_name="0.0.0.0", server_port=7860)