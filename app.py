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

# Main app password (already existed)
APP_PASSWORD_HASH = "5e884898da28047151d0e56f8dc6292773603d0d6aabbdd62a11ef721d1542d8"  # "password"

# NEW: Separate admin password to unlock sensitive settings
# Change this to something strong and keep it secret
ADMIN_PASSWORD_PLAIN = "Hardik@123"  # ← CHANGE THIS
ADMIN_PASSWORD_HASH = hashlib.sha256(ADMIN_PASSWORD_PLAIN.encode()).hexdigest()

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
                if pid.strip():
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

# ─── Config Helpers ──────────────────────────────────────
def load_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE) as f:
            return json.load(f)
    return {}

def save_config(cfg):
    with open(CONFIG_FILE, 'w') as f:
        json.dump(cfg, f)

# ─── Main App ────────────────────────────────────────────
def main_app():
    with st.sidebar:
        st.title("⚙️ Controls")

        # Profile
        st.subheader("👤 Profile")
        username = st.text_input("Username", value="hardik")
        if st.button("Set User"):
            st.session_state['user_id'] = get_or_create_user(username)
            # Reset current session when changing user
            if 'current_session' in st.session_state:
                del st.session_state['current_session']
            st.success(f"User set to: {username}")
            st.rerun()

        if 'user_id' not in st.session_state:
            st.warning("Please set a username first")
            st.stop()

        user_id = st.session_state['user_id']

        # ── ADMIN SECTION ── PROTECTED ────────────────────────────────
        st.subheader("🛡️ Admin / Developer Settings")
        admin_pass = st.text_input("Admin password", type="password", key="admin_unlock")

        if admin_pass:
            if hashlib.sha256(admin_pass.encode()).hexdigest() == ADMIN_PASSWORD_HASH:
                st.success("Admin access granted", icon="🔓")

                # Telegram Bot control – only visible after correct admin password
                st.subheader("📱 Telegram Bot")
                config = load_config()
                token_input = st.text_input(
                    "Bot Token",
                    value=config.get('telegram_token', ''),
                    type="password",
                    key="bot_token_protected"
                )
                if st.button("Save Token", key="save_token_protected"):
                    config['telegram_token'] = token_input.strip()
                    save_config(config)
                    st.success("Token saved")

                bot_running = check_bot_status()
                st.write("Status: " + ("🟢 Running" if bot_running else "🔴 Stopped"))

                col1, col2 = st.columns(2)
                with col1:
                    if not bot_running and st.button("Start Bot", key="start_bot_protected"):
                        if token_input.strip():
                            start_bot()
                            st.success("Bot starting...")
                            st.rerun()
                        else:
                            st.error("Bot token is required")
                with col2:
                    if bot_running and st.button("Stop Bot", key="stop_bot_protected"):
                        stop_bot()
                        st.success("Bot stopped")
                        st.rerun()

                st.divider()

                # NVIDIA API – only visible after correct admin password
                st.subheader("🔑 NVIDIA API")
                api_key_input = st.text_input(
                    "NVIDIA API Key",
                    value=config.get('api_key', ''),
                    type="password",
                    key="nvidia_key_protected"
                )
                if st.button("Save Key", key="save_key_protected"):
                    config['api_key'] = api_key_input.strip()
                    save_config(config)
                    st.success("API key saved")

            else:
                st.error("Incorrect admin password")
        else:
            st.info("Enter admin password to manage bot token & API key", icon="🔐")

        st.divider()

        # Model selection – left visible (not sensitive)
        config = load_config()
        model = st.selectbox("Model", [
            "meta/llama3-70b-instruct",
            "meta/llama3-8b-instruct",
            "mistralai/mixtral-8x7b-instruct-v0.1",
            "google/gemma-7b-it"
        ], index=0)

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
            if 'current_session' in st.session_state:
                update_session_prompt(st.session_state['current_session'], prompt)
                supabase.table("sessions").update({"persona_name": selected}).eq("id", st.session_state['current_session']).execute()
                st.session_state['system_prompt'] = prompt
                st.success(f"Persona changed to **{selected}**")
                st.rerun()

        st.divider()

        # Sessions – improved listing
        st.subheader("📁 Your Sessions")

        if st.button("➕ New Session"):
            sid = create_session(user_id)
            st.session_state['current_session'] = sid
            st.session_state['messages'] = []
            st.session_state['system_prompt'] = "You are a helpful assistant."
            st.rerun()

        sessions = get_sessions(user_id)

        if sessions:
            for s in sessions:
                title = s['title'] or f"Chat {s['created_at'][:16]}"
                is_active = 'current_session' in st.session_state and st.session_state['current_session'] == s['id']

                cols = st.columns([6, 1])
                with cols[0]:
                    if st.button(
                        f"{'→ ' if is_active else ''}{title}",
                        key=f"session_load_{s['id']}",
                        use_container_width=True,
                        type="primary" if is_active else "secondary"
                    ):
                        st.session_state['current_session'] = s['id']
                        st.session_state['messages'] = load_messages(s['id'])
                        st.session_state['system_prompt'] = get_session_prompt(s['id'])
                        st.rerun()

                with cols[1]:
                    if st.button("🗑", key=f"session_del_{s['id']}"):
                        delete_session(s['id'])
                        if 'current_session' in st.session_state and st.session_state['current_session'] == s['id']:
                            del st.session_state['current_session']
                        st.rerun()

        if 'current_session' in st.session_state:
            if st.button("🧹 Clear current messages"):
                clear_current_session(st.session_state['current_session'])
                st.session_state['messages'] = []
                st.success("Chat cleared")
                st.rerun()

    # ─── Main Area ───────────────────────────────────────────
    st.title("🤖 NVIDIA AI Chat")

    if 'current_session' not in st.session_state:
        st.session_state['current_session'] = create_session(st.session_state['user_id'])
        st.session_state['messages'] = []
        st.session_state['system_prompt'] = "You are a helpful assistant."

    if 'messages' not in st.session_state:
        st.session_state['messages'] = load_messages(st.session_state['current_session'])

    if 'system_prompt' not in st.session_state:
        st.session_state['system_prompt'] = get_session_prompt(st.session_state['current_session'])

    # System prompt view/edit
    with st.expander("System Prompt"):
        prompt_edit = st.text_area("Prompt", value=st.session_state['system_prompt'], height=120)
        if st.button("Update Prompt"):
            update_session_prompt(st.session_state['current_session'], prompt_edit)
            st.session_state['system_prompt'] = prompt_edit
            st.success("Updated")
            st.rerun()

    # Messages
    for msg in st.session_state['messages']:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            st.caption(str(msg["timestamp"]))

    # Input
    if prompt := st.chat_input("Message..."):
        if not config.get('api_key'):
            st.error("NVIDIA API key not set (admin section)")
        else:
            with st.chat_message("user"):
                st.markdown(prompt)

            save_message(st.session_state['current_session'], 'user', prompt)
            st.session_state['messages'] = load_messages(st.session_state['current_session'])

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    api_messages = [{"role": "system", "content": st.session_state['system_prompt']}]
                    for m in st.session_state['messages']:
                        if m["role"] in ['user', 'assistant']:
                            api_messages.append({"role": m["role"], "content": m["content"]})

                    try:
                        response, usage = send_to_nvidia(config['api_key'], model, api_messages)
                        st.markdown(response)

                        save_message(st.session_state['current_session'], 'assistant', response)
                        st.session_state['messages'] = load_messages(st.session_state['current_session'])

                        if usage:
                            pt = usage.get('prompt_tokens', '?')
                            ct = usage.get('completion_tokens', '?')
                            st.caption(f"Tokens: {pt} + {ct}")

                    except Exception as e:
                        st.error(f"Error: {e}")
                        save_message(st.session_state['current_session'], 'system', str(e))

            st.rerun()

# ─── Entry ───────────────────────────────────────────────
if __name__ == "__main__":
    if 'authenticated' not in st.session_state:
        st.session_state['authenticated'] = False

    if not st.session_state['authenticated']:
        login_screen()
    else:
        main_app()
