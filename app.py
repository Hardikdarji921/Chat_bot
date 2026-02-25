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
import io  # For in-memory file buffers

# ─── NEW: For improved PDF generation with styling ───────
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas

# ─── Configuration ───────────────────────────────────────
# File where Telegram token and NVIDIA API key are stored
CONFIG_FILE = 'config.json'
# Name of the Telegram bot script to start/stop
BOT_SCRIPT = 'telegram_bot.py'

# Main login password hash (for the Streamlit app dashboard)
APP_PASSWORD_HASH = "5e884898da28047151d0e56f8dc6292773603d0d6aabbdd62a11ef721d1542d8"  # hash of "password"

# Separate admin password hash to protect sensitive settings
ADMIN_PASSWORD_PLAIN = "Hardik@123"  # ← CHANGE THIS TO A STRONG PASSWORD
ADMIN_PASSWORD_HASH = hashlib.sha256(ADMIN_PASSWORD_PLAIN.encode()).hexdigest()

# Supabase connection details (public anon key - safe for client-side)
SUPABASE_URL = "https://phonjftgqkutfeigdrts.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InBob25qZnRncWt1dGZlaWdkcnRzIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzE4Mjg5OTAsImV4cCI6MjA4NzQwNDk5MH0.w4ZHZEQXaYHCDMraFRsnRRM1WAfKRhXm25YwB6g33XM"

# Initialize Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ─── Page Config ─────────────────────────────────────────
# Set global Streamlit page settings
st.set_page_config(
    page_title="NVIDIA AI Chat v5.2",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Supabase Helper Functions ───────────────────────────
# Get or create user record based on username
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

# Fetch all sessions for a given user, sorted newest first
def get_sessions(user_id: int):
    resp = supabase.table("sessions") \
        .select("id, title, created_at, system_prompt, persona_name") \
        .eq("user_id", user_id) \
        .order("created_at", desc=True) \
        .execute()
    return resp.data

# Create a new chat session for the user
def create_session(user_id: int, title="New Chat"):
    data = {
        "user_id": user_id,
        "title": title,
        "system_prompt": "You are a helpful assistant.",
        "persona_name": "Default Assistant"
    }
    resp = supabase.table("sessions").insert(data).execute()
    return resp.data[0]["id"]

# Load all messages for a specific session
def load_messages(session_id: int):
    resp = supabase.table("messages") \
        .select("id, role, content, timestamp") \
        .eq("session_id", session_id) \
        .order("timestamp", desc=False) \
        .execute()
    return resp.data

# Save a new message (user or assistant) to the database
def save_message(session_id: int, role: str, content: str):
    supabase.table("messages").insert({
        "session_id": session_id,
        "role": role,
        "content": content
    }).execute()

# Update the content of an existing message
def update_message(msg_id: int, new_content: str):
    supabase.table("messages").update({"content": new_content}).eq("id", msg_id).execute()

# Delete a single message
def delete_message(msg_id: int):
    supabase.table("messages").delete().eq("id", msg_id).execute()

# Delete all messages after a certain point (used after editing)
def truncate_messages(session_id: int, msg_id: int):
    supabase.table("messages").delete() \
        .eq("session_id", session_id) \
        .gt("id", msg_id) \
        .execute()

# Update the system prompt of a session
def update_session_prompt(session_id: int, prompt: str):
    supabase.table("sessions").update({"system_prompt": prompt}).eq("id", session_id).execute()

# Update the display title of a session
def update_session_title(session_id: int, title: str):
    supabase.table("sessions").update({"title": title}).eq("id", session_id).execute()

# Delete an entire session (messages are deleted via cascade or manually)
def delete_session(session_id: int):
    supabase.table("sessions").delete().eq("id", session_id).execute()

# Get only the system prompt of a session
def get_session_prompt(session_id: int):
    resp = supabase.table("sessions").select("system_prompt").eq("id", session_id).execute()
    return resp.data[0]["system_prompt"] if resp.data else "You are a helpful assistant."

# Clear all messages in the current session
def clear_current_session(session_id: int):
    supabase.table("messages").delete().eq("session_id", session_id).execute()

# Send chat completion request to NVIDIA API
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

# Check if telegram_bot.py process is currently running
def check_bot_status():
    try:
        result = subprocess.run(['pgrep', '-f', 'telegram_bot.py'], capture_output=True, text=True)
        return result.returncode == 0
    except:
        return False

# Start the Telegram bot in a new process
def start_bot():
    subprocess.Popen(['python', BOT_SCRIPT], start_new_session=True)

# Gracefully stop the Telegram bot process
def stop_bot():
    try:
        result = subprocess.run(['pgrep', '-f', 'telegram_bot.py'], capture_output=True, text=True)
        if result.returncode == 0:
            for pid in result.stdout.strip().split('\n'):
                if pid.strip():
                    os.kill(int(pid), signal.SIGTERM)
    except:
        pass

# ─── Auto-title generation functions ─────────────────────
def auto_generate_simple_title(session_id: int):
    """Simple version: use first user message (zero cost)"""
    messages = load_messages(session_id)
    if not messages:
        return None
    first_user_msg = next(
        (m['content'].strip() for m in messages if m['role'] == 'user'),
        None
    )
    if not first_user_msg:
        return None
    title = first_user_msg.replace("\n", " ").strip()
    if len(title) > 60:
        title = title[:57] + "..."
    if title:
        update_session_title(session_id, title[:70])
        return title
    return None

def auto_generate_llm_title(session_id: int, api_key: str, model: str):
    """Better version: ask LLM to summarize (after some messages)"""
    messages = load_messages(session_id)
    if len(messages) < 4:
        return None
    recent = messages[-6:]
    context = "\n".join([f"{m['role']}: {m['content'][:150]}" for m in recent])
    prompt_messages = [
        {
            "role": "system",
            "content": "Create a very short, clear session title (4–8 words max). "
                       "Focus on the main topic. No quotes, no explanations, no punctuation at the end."
        },
        {
            "role": "user",
            "content": f"Summarize this conversation into a short title:\n\n{context}"
        }
    ]
    try:
        title_raw, _ = send_to_nvidia(api_key, model, prompt_messages)
        title = title_raw.strip().strip('"').strip("'").strip()
        words = title.split()
        if 3 <= len(words) <= 10 and len(title) <= 70:
            update_session_title(session_id, title)
            return title
    except Exception:
        pass
    return None

# ─── Generate readable text content for fallback/export ──
def generate_session_text(session_id: int):
    """Builds a readable plain-text representation of the entire session"""
    session = supabase.table("sessions").select("title, system_prompt, persona_name, created_at").eq("id", session_id).single().execute().data
    messages = load_messages(session_id)

    lines = []
    lines.append("═══════════════════════════════════════════════════════════════")
    lines.append(f"Session ID: {session_id}")
    lines.append(f"Title: {session['title'] or 'Untitled Chat'}")
    lines.append(f"Created: {session['created_at']}")
    lines.append(f"Persona: {session['persona_name']}")
    lines.append(f"System Prompt:\n{session['system_prompt']}")
    lines.append("═══════════════════════════════════════════════════════════════\n")

    for msg in messages:
        role = msg['role'].upper()
        ts = msg['timestamp']
        content = msg['content'].strip()
        lines.append(f"[{ts}] {role}:")
        lines.append(content)
        lines.append("─" * 70 + "\n")

    return "\n".join(lines)

# ─── Generate Markdown content for .md export ────────────
def generate_session_markdown(session_id: int):
    """Creates Markdown-formatted content of the session with images preserved"""
    session = supabase.table("sessions").select("title, system_prompt, persona_name, created_at").eq("id", session_id).single().execute().data
    messages = load_messages(session_id)

    lines = []
    lines.append(f"# {session['title'] or 'Untitled Chat'}")
    lines.append("")
    lines.append(f"**Session ID:** {session_id}")
    lines.append(f"**Created:** {session['created_at']}")
    lines.append(f"**Persona:** {session['persona_name']}")
    lines.append("")
    lines.append("**System Prompt:**")
    lines.append(f"```text")
    lines.append(session['system_prompt'])
    lines.append("```")
    lines.append("")
    lines.append("---")
    lines.append("")

    for msg in messages:
        role = msg['role'].capitalize()
        ts = msg['timestamp']
        content = msg['content'].strip()

        # Preserve images if content contains image URLs or markdown images
        if "http" in content and any(ext in content.lower() for ext in [".png", ".jpg", ".jpeg", ".gif", ".webp"]):
            lines.append(f"**[{ts}] {role}**  \n{content}")
        else:
            lines.append(f"**[{ts}] {role}**")
            lines.append(content.replace("\n", "  \n"))  # preserve line breaks

        lines.append("")
        lines.append("---")
        lines.append("")

    return "\n".join(lines)

# ─── NEW: Improved PDF generation with colors, fonts, header/footer ──
def generate_session_pdf_bytes(session_id: int):
    """Creates a styled PDF with header, footer, colors, and image placeholders"""
    session = supabase.table("sessions").select("title, system_prompt, persona_name, created_at").eq("id", session_id).single().execute().data
    messages = load_messages(session_id)

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=72
    )

    styles = getSampleStyleSheet()

    # Custom styles for better look
    title_style = ParagraphStyle(
        'Title',
        parent=styles['Title'],
        fontName='Helvetica-Bold',
        fontSize=18,
        textColor=colors.darkblue,
        spaceAfter=12
    )

    subtitle_style = ParagraphStyle(
        'Subtitle',
        parent=styles['Heading2'],
        fontName='Helvetica',
        fontSize=12,
        textColor=colors.grey,
        spaceAfter=6
    )

    role_style = ParagraphStyle(
        'Role',
        parent=styles['Heading3'],
        fontName='Helvetica-Bold',
        fontSize=12,
        textColor=colors.navy,
        spaceBefore=12,
        spaceAfter=6
    )

    content_style = ParagraphStyle(
        'Content',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        spaceAfter=8
    )

    timestamp_style = ParagraphStyle(
        'Timestamp',
        parent=styles['Italic'],
        fontName='Helvetica-Oblique',
        fontSize=9,
        textColor=colors.grey,
        spaceAfter=4
    )

    story = []

    # Header (on first page)
    story.append(Paragraph(f"{session['title'] or 'Chat Session'}", title_style))
    story.append(Paragraph(f"Session ID: {session_id} • Created: {session['created_at']}", subtitle_style))
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"Persona: {session['persona_name']}", subtitle_style))
    story.append(Spacer(1, 24))

    story.append(Paragraph("System Prompt:", role_style))
    story.append(Paragraph(session['system_prompt'].replace("\n", "<br/>"), content_style))
    story.append(Spacer(1, 36))

    # Messages
    for msg in messages:
        ts = str(msg['timestamp'])
        role = msg['role'].upper()
        content = msg['content'].replace("\n", "<br/>")

        # Detect if message contains image URL
        image_url = None
        if "http" in content and any(ext in content.lower() for ext in [".png", ".jpg", ".jpeg", ".gif", ".webp"]):
            # Very basic URL extraction - improve if needed
            start = content.find("http")
            end = content.find(" ", start) if " " in content[start:] else len(content)
            image_url = content[start:end].strip()

        story.append(Paragraph(f"[{ts}] {role}", role_style))
        story.append(Paragraph(content, content_style))

        # Placeholder for image in PDF (reportlab doesn't embed remote images easily)
        if image_url:
            story.append(Paragraph(f"[Image included in original chat: {image_url}]", timestamp_style))
            story.append(Spacer(1, 12))

        story.append(Spacer(1, 18))

    # Build PDF
    doc.build(story)

    # Footer (page number) - added via onPage
    def add_page_number(canvas, doc):
        page_num = canvas.getPageNumber()
        canvas.setFont("Helvetica", 9)
        canvas.setFillColor(colors.grey)
        canvas.drawRightString(
            doc.rightMargin + doc.width,
            doc.bottomMargin - 0.5 * inch,
            f"Page {page_num} • Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        )

    # Re-build with footer
    buffer.seek(0)
    doc.build(story, onFirstPage=add_page_number, onLaterPages=add_page_number)

    buffer.seek(0)
    return buffer

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

        config = load_config()
        model = st.selectbox("Model", [
            "meta/llama3-70b-instruct",
            "meta/llama3-8b-instruct",
            "mistralai/mixtral-8x7b-instruct-v0.1",
            "google/gemma-7b-it"
        ], index=0)

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
                title = s['title'] or f"Chat {s['created_at'][:16].replace('T', ' ')}"
                is_active = 'current_session' in st.session_state and st.session_state['current_session'] == s['id']

                cols = st.columns([7, 1])
                with cols[0]:
                    if st.button(
                        f"{'→ ' if is_active else ''}{title}",
                        key=f"load_session_{s['id']}",
                        use_container_width=True,
                        type="primary" if is_active else "secondary"
                    ):
                        st.session_state['current_session'] = s['id']
                        st.session_state['messages'] = load_messages(s['id'])
                        st.session_state['system_prompt'] = get_session_prompt(s['id'])
                        st.rerun()

                with cols[1]:
                    if st.button("🗑", key=f"delete_session_{s['id']}"):
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

    # ─── Main Chat Area ──────────────────────────────────────
    st.title("🤖 NVIDIA AI Chat")

    if 'current_session' not in st.session_state:
        st.session_state['current_session'] = create_session(st.session_state['user_id'])
        st.session_state['messages'] = []
        st.session_state['system_prompt'] = "You are a helpful assistant."

    if 'messages' not in st.session_state:
        st.session_state['messages'] = load_messages(st.session_state['current_session'])

    if 'system_prompt' not in st.session_state:
        st.session_state['system_prompt'] = get_session_prompt(st.session_state['current_session'])

    with st.expander("System Prompt (edit if needed)"):
        prompt_edit = st.text_area("Prompt", value=st.session_state['system_prompt'], height=120)
        if st.button("Update Prompt"):
            update_session_prompt(st.session_state['current_session'], prompt_edit)
            st.session_state['system_prompt'] = prompt_edit
            st.success("System prompt updated")
            st.rerun()

    for msg in st.session_state['messages']:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            st.caption(str(msg["timestamp"]))

    # ─── Export Section: PDF & Markdown with improved PDF styling ──
    if 'current_session' in st.session_state:
        st.markdown("---")
        st.subheader("Export / Download Session")

        col_pdf, col_md = st.columns(2)

        with col_pdf:
            if st.button("📄 Download as Styled PDF", use_container_width=True):
                pdf_buffer = generate_session_pdf_bytes(st.session_state['current_session'])

                session_title = supabase.table("sessions").select("title").eq("id", st.session_state['current_session']).single().execute().data['title']
                safe_title = (session_title or "Chat_Session").replace(" ", "_").replace("/", "-")[:50]
                filename = f"{safe_title}_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"

                st.download_button(
                    label="Click to download PDF",
                    data=pdf_buffer,
                    file_name=filename,
                    mime="application/pdf",
                    key="download_pdf_styled"
                )
                st.success(f"Styled PDF ready: {filename}")

        with col_md:
            if st.button("📝 Download as Markdown (.md)", use_container_width=True):
                md_content = generate_session_markdown(st.session_state['current_session'])

                session_title = supabase.table("sessions").select("title").eq("id", st.session_state['current_session']).single().execute().data['title']
                safe_title = (session_title or "Chat_Session").replace(" ", "_").replace("/", "-")[:50]
                filename = f"{safe_title}_{datetime.now().strftime('%Y%m%d_%H%M')}.md"

                st.download_button(
                    label="Click to download .md",
                    data=md_content,
                    file_name=filename,
                    mime="text/markdown",
                    key="download_md"
                )
                st.success(f"Markdown ready: {filename}")

    # ─── Chat Input + Response Generation ────────────────────
    if prompt := st.chat_input("Message..."):
        if not config.get('api_key'):
            st.error("NVIDIA API key not set (admin section)")
        else:
            with st.chat_message("user"):
                st.markdown(prompt)

            save_message(st.session_state['current_session'], 'user', prompt)
            st.session_state['messages'] = load_messages(st.session_state['current_session'])

            if len(st.session_state['messages']) == 1:
                simple_title = auto_generate_simple_title(st.session_state['current_session'])
                if simple_title:
                    st.toast(f"Session titled: {simple_title}", icon="✨")
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

                        if len(st.session_state['messages']) >= 4 and config.get('api_key'):
                            llm_title = auto_generate_llm_title(
                                st.session_state['current_session'],
                                config['api_key'],
                                model
                            )
                            if llm_title:
                                st.toast(f"Improved title: {llm_title}", icon="✨")
                                st.session_state['messages'] = load_messages(st.session_state['current_session'])

                        if usage:
                            pt = usage.get('prompt_tokens', '?')
                            ct = usage.get('completion_tokens', '?')
                            st.caption(f"Tokens: {pt} + {ct}")

                    except Exception as e:
                        st.error(f"Error: {e}")
                        save_message(st.session_state['current_session'], 'system', str(e))

            st.session_state['messages'] = load_messages(st.session_state['current_session'])
            st.rerun()

# ─── Entry Point ─────────────────────────────────────────
if __name__ == "__main__":
    if 'authenticated' not in st.session_state:
        st.session_state['authenticated'] = False

    if not st.session_state['authenticated']:
        login_screen()
    else:
        main_app()
