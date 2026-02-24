import streamlit as st
import requests
import json
import os
import hashlib
import subprocess
import signal
from datetime import datetime
from supabase import create_client, Client
from duckduckgo_search import DDGS

# ─── Configuration ───────────────────────────────────────
CONFIG_FILE = 'config.json'
BOT_SCRIPT = 'telegram_bot.py'
APP_PASSWORD_HASH = "5e884898da28047151d0e56f8dc6292773603d0d6aabbdd62a11ef721d1542d8"  # "password"

SUPABASE_URL = "https://phonjftgqkutfeigdrts.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InBob25qZnRncWt1dGZlaWdkcnRzIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzE4Mjg5OTAsImV4cCI6MjA4NzQwNDk5MH0.w4ZHZEQXaYHCDMraFRsnRRM1WAfKRhXm25YwB6g33XM"

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ─── Page Config ─────────────────────────────────────────
st.set_page_config(
    page_title="NVIDIA AI Chat v5.2",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Supabase Helpers ────────────────────────────────────
def get_or_create_user(username: str, telegram_id=None):
    resp = supabase.table("users").select("id").eq("username", username).execute()
    if resp.data:
        uid = resp.data[0]["id"]
        if telegram_id:
            supabase.table("users").update({"telegram_id": telegram_id}).eq("id", uid).execute()
        return uid

    data = {"username": username}
    if telegram_id: data["telegram_id"] = telegram_id
    resp = supabase.table("users").insert(data).execute()
    return resp.data[0]["id"]

def get_sessions(user_id: int):
    resp = supabase.table("sessions") \
        .select("id, title, created_at, system_prompt, persona_name") \
        .eq("user_id", user_id) \
        .order("created_at", desc=True) \
        .execute()
    return resp.data

def create_session(user_id: int, title="New Chat"):
    data = {
        "user_id": user_id,
        "title": title,
        "system_prompt": "You are a helpful assistant.",
        "persona_name": "Default Assistant"
    }
    resp = supabase.table("sessions").insert(data).execute()
    return resp.data[0]["id"]

def load_messages(session_id: int):
    resp = supabase.table("messages") \
        .select("id, role, content, timestamp") \
        .eq("session_id", session_id) \
        .order("timestamp", desc=False) \
        .execute()
    return resp.data

def save_message(session_id: int, role: str, content: str):
    supabase.table("messages").insert({
        "session_id": session_id,
        "role": role,
        "content": content
    }).execute()

def update_message(msg_id: int, new_content: str):
    supabase.table("messages").update({"content": new_content}).eq("id", msg_id).execute()

def delete_message(msg_id: int):
    supabase.table("messages").delete().eq("id", msg_id).execute()

def truncate_messages(session_id: int, msg_id: int):
    supabase.table("messages").delete() \
        .eq("session_id", session_id) \
        .gt("id", msg_id) \
        .execute()

def update_session_prompt(session_id: int, prompt: str):
    supabase.table("sessions").update({"system_prompt": prompt}).eq("id", session_id).execute()

def update_session_title(session_id: int, title: str):
    supabase.table("sessions").update({"title": title}).eq("id", session_id).execute()

def delete_session(session_id: int):
    supabase.table("sessions").delete().eq("id", session_id).execute()

def rename_session(session_id: int, new_title: str):
    supabase.table("sessions").update({"title": new_title}).eq("id", session_id).execute()

def get_session_prompt(session_id: int):
    resp = supabase.table("sessions").select("system_prompt").eq("id", session_id).execute()
    return resp.data[0]["system_prompt"] if resp.data else "You are a helpful assistant."

def clear_current_session(session_id: int):
    supabase.table("messages").delete().eq("session_id", session_id).execute()

def send_to_nvidia(api_key, model, messages):
    url = "https://integrate.api.nvidia.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": model, "messages": messages, "max_tokens": 2048, "temperature": 0.7}
    resp = requests.post(url, headers=headers, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    content = data['choices'][0]['message']['content']
    usage = data.get('usage', {})
    return content, usage

def read_file_content(uploaded_file):
    try:
        return f"\n--- {uploaded_file.name} ---\n{uploaded_file.getvalue().decode('utf-8')}\n--- End ---\n"
    except:
        return f"\n--- Error reading {uploaded_file.name} ---\n"

def check_bot_status():
    try:
        result = subprocess.run(['pgrep', '-f', 'telegram_bot.py'], capture_output=True, text=True)
        return result.returncode == 0
    except:
        return False

def start_bot():
    subprocess.Popen(['python', BOT_SCRIPT], start_new_session=True)

def stop_bot():
    try:
        result = subprocess.run(['pgrep', '-f', 'telegram_bot.py'], capture_output=True, text=True)
        if result.returncode == 0:
            for pid in result.stdout.strip().split('\n'):
                os.kill(int(pid), signal.SIGTERM)
    except:
        pass

def trigger_ai_response(session_id, api_key, model, system_prompt):
    messages = load_messages(session_id)
    api_messages = [{"role": "system", "content": system_prompt}]
    for m in messages:
        if m["role"] in ['user', 'assistant']:
            api_messages.append({"role": m["role"], "content": m["content"]})

    if not messages or messages[-1]["role"] != 'user':
        return False, None

    try:
        response, usage = send_to_nvidia(api_key, model, api_messages)
        save_message(session_id, 'assistant', response)
        return True, (response, usage)
    except Exception as e:
        return False, str(e)

# ─── Login ───────────────────────────────────────────────
def login_screen():
    st.title("🔒 Login")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if hashlib.sha256(password.encode()).hexdigest() == APP_PASSWORD_HASH:
            st.session_state['authenticated'] = True
            st.rerun()
        else:
            st.error("Wrong password")

# ─── Main App ────────────────────────────────────────────
def main_app():
    with st.sidebar:
        st.title("⚙️ Controls")

        # Profile
        st.subheader("👤 Profile")
        username = st.text_input("Username", value="default_user")
        if st.button("Set User"):
            st.session_state['user_id'] = get_or_create_user(username)
            st.success(f"User: {username}")
            st.rerun()

        if 'user_id' not in st.session_state:
            st.warning("Set username first")
            st.stop()

        user_id = st.session_state['user_id']

        # Telegram bot control
        st.subheader("📱 Telegram Bot")
        config = load_config()
        token_input = st.text_input("Bot Token", value=config.get('telegram_token',''), type="password")
        if st.button("Save Token"):
            config['telegram_token'] = token_input
            save_config(config)
            st.success("Saved")

        bot_running = check_bot_status()
        st.write("Status: " + ("🟢 Running" if bot_running else "🔴 Stopped"))

        col1, col2 = st.columns(2)
        with col1:
            if not bot_running and st.button("Start Bot"):
                if token_input:
                    start_bot()
                    st.success("Starting...")
                    st.rerun()
                else:
                    st.error("Token required")
        with col2:
            if bot_running and st.button("Stop Bot"):
                stop_bot()
                st.success("Stopped")
                st.rerun()

        st.divider()

        # API & Model
        st.subheader("🔑 API")
        api_key = st.text_input("NVIDIA API Key", value=config.get('api_key',''), type="password")
        if st.button("Save Key"):
            config['api_key'] = api_key
            save_config(config)
            st.success("Saved")

        model = st.selectbox("Model", [
            "meta/llama3-70b-instruct",
            "meta/llama3-8b-instruct",
            "mistralai/mixtral-8x7b-instruct-v0.1",
            "google/gemma-7b-it"
        ])

        # Personas
        st.subheader("🎭 Persona")
        personas = {
            "Default Assistant": "You are a helpful, concise and friendly assistant.",
            "Code Expert": "You are a senior software engineer. Write clean, well-commented code. Prefer Python.",
            "Sarcastic Friend": "You are sarcastic, witty, tease lightly, but still helpful.",
            "Story Teller": "You are a creative storyteller. Use vivid language, build suspense.",
            "Ultra Concise": "Answer in 1–2 short sentences. No fluff."
        }

        selected = st.selectbox("Select persona", list(personas.keys()))
        if st.button("Apply"):
            prompt = personas[selected]
            update_session_prompt(st.session_state['current_session'], prompt)
            supabase.table("sessions").update({"persona_name": selected}).eq("id", st.session_state['current_session']).execute()
            st.session_state['system_prompt'] = prompt
            st.success(f"Persona: **{selected}**")
            st.rerun()

        st.divider()

        # Sessions
        st.subheader("📁 Sessions")
        if st.button("➕ New Session"):
            sid = create_session(user_id)
            st.session_state['current_session'] = sid
            st.session_state['messages'] = []
            st.session_state['system_prompt'] = "You are a helpful assistant."
            st.rerun()

        sessions = get_sessions(user_id)
        if sessions:
            opts = {f"{s['title']} ({s.get('persona_name','')}) — ID:{s['id']}": s['id'] for s in sessions}
            choice = st.selectbox("Open session", list(opts.keys()))
            if st.button("Load"):
                sid = opts[choice]
                st.session_state['current_session'] = sid
                st.session_state['messages'] = load_messages(sid)
                st.session_state['system_prompt'] = next(s["system_prompt"] for s in sessions if s["id"] == sid)
                st.rerun()

        if st.button("🗑️ Delete current"):
            if 'current_session' in st.session_state:
                delete_session(st.session_state['current_session'])
                st.session_state.pop('current_session', None)
                st.rerun()

        if st.button("🧹 Clear messages"):
            if 'current_session' in st.session_state:
                clear_current_session(st.session_state['current_session'])
                st.session_state['messages'] = []
                st.success("Chat cleared")
                st.rerun()

        st.divider()

    # ─── Main Area ───────────────────────────────────────────
    st.title("🤖 NVIDIA AI Chat")

    if 'current_session' not in st.session_state:
        st.session_state['current_session'] = create_session(user_id)
        st.session_state['messages'] = []
        st.session_state['system_prompt'] = "You are a helpful assistant."

    if 'messages' not in st.session_state:
        st.session_state['messages'] = load_messages(st.session_state['current_session'])

    if 'system_prompt' not in st.session_state:
        st.session_state['system_prompt'] = get_session_prompt(st.session_state['current_session'])

    # Regenerate after edit / delete / regenerate
    if st.session_state.get('trigger_ai_after_edit', False):
        st.session_state['trigger_ai_after_edit'] = False
        if 'api_key' in locals() and api_key:
            with st.spinner("Regenerating..."):
                success, result = trigger_ai_response(
                    st.session_state['current_session'],
                    api_key,
                    model,
                    st.session_state['system_prompt']
                )
                if success:
                    response, usage = result
                    st.session_state['messages'] = load_messages(st.session_state['current_session'])
                    st.success("Regenerated")
                else:
                    st.error(result)
                st.rerun()

    # System prompt view/edit
    with st.expander("System Prompt"):
        prompt_edit = st.text_area("Prompt", value=st.session_state['system_prompt'], height=120)
        if st.button("Update Prompt"):
            update_session_prompt(st.session_state['current_session'], prompt_edit)
            st.session_state['system_prompt'] = prompt_edit
            st.success("Updated")
            st.rerun()

    # Messages
    editing_id = st.session_state.get('editing_msg_id', None)

    for msg in st.session_state['messages']:
        mid = msg["id"]
        role = msg["role"]
        content = msg["content"]
        ts = msg["timestamp"]

        if editing_id == mid:
            with st.chat_message(role):
                st.write(f"Editing — {ts}")
                edit_text = st.text_area("Edit", value=content, height=140, key=f"edit_{mid}")
                c1, c2 = st.columns(2)
                with c1:
                    if st.button("Save & Regenerate", key=f"save_{mid}"):
                        update_message(mid, edit_text)
                        truncate_messages(st.session_state['current_session'], mid)
                        st.session_state['messages'] = load_messages(st.session_state['current_session'])
                        st.session_state['editing_msg_id'] = None
                        st.session_state['trigger_ai_after_edit'] = True
                        st.rerun()
                with c2:
                    if st.button("Cancel", key=f"cancel_{mid}"):
                        st.session_state['editing_msg_id'] = None
                        st.rerun()
        else:
            with st.chat_message(role):
                st.markdown(content)
                st.caption(ts)

                if role == "assistant":
                    if st.button("🔄 Regenerate", key=f"regen_{mid}", help="New response"):
                        delete_message(mid)
                        st.session_state['messages'] = load_messages(st.session_state['current_session'])
                        st.session_state['trigger_ai_after_edit'] = True
                        st.rerun()

                c1, c2 = st.columns([1,5])
                with c1:
                    if st.button("✏", key=f"editbtn_{mid}"):
                        st.session_state['editing_msg_id'] = mid
                        st.rerun()
                with c2:
                    if st.button("🗑", key=f"del_{mid}"):
                        before = [m for m in st.session_state['messages'] if m["id"] < mid]
                        delete_message(mid)
                        truncate_messages(st.session_state['current_session'], mid)
                        st.session_state['messages'] = load_messages(st.session_state['current_session'])
                        if before and before[-1]["role"] == "user":
                            st.session_state['trigger_ai_after_edit'] = True
                        st.success("Deleted")
                        st.rerun()

    # Input
    if prompt := st.chat_input("Message..."):
        if not api_key:
            st.error("API key missing")
        else:
            with st.chat_message("user"):
                st.write(prompt)

            full_prompt = prompt
            save_message(st.session_state['current_session'], 'user', full_prompt)
            st.session_state['messages'] = load_messages(st.session_state['current_session'])

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    api_messages = [{"role": "system", "content": st.session_state['system_prompt']}]
                    for m in st.session_state['messages']:
                        if m["role"] in ['user', 'assistant']:
                            api_messages.append({"role": m["role"], "content": m["content"]})

                    try:
                        response, usage = send_to_nvidia(api_key, model, api_messages)
                        st.markdown(response)

                        save_message(st.session_state['current_session'], 'assistant', response)
                        st.session_state['messages'] = load_messages(st.session_state['current_session'])

                        # Show usage
                        if usage:
                            pt = usage.get('prompt_tokens', '?')
                            ct = usage.get('completion_tokens', '?')
                            total = pt + ct if isinstance(pt,int) and isinstance(ct,int) else "?"
                            cost_est = round((total / 1_000_000) * 0.50, 5) if isinstance(total, int) else "?"
                            st.caption(f"Tokens: {pt} + {ct} = {total}  •  ~${cost_est}")

                    except Exception as e:
                        st.error(f"Error: {e}")
                        save_message(st.session_state['current_session'], 'system', str(e))

# ─── Helpers ─────────────────────────────────────────────
def load_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE) as f:
            return json.load(f)
    return {}

def save_config(cfg):
    with open(CONFIG_FILE, 'w') as f:
        json.dump(cfg, f)

# ─── Entry ───────────────────────────────────────────────
if __name__ == "__main__":
    if 'authenticated' not in st.session_state:
        st.session_state['authenticated'] = False

    if not st.session_state['authenticated']:
        login_screen()
    else:
        main_app()