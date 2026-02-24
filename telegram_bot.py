import logging
import json
import os
from datetime import datetime
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from supabase import create_client, Client
from duckduckgo_search import DDGS

# ─── Configuration ───────────────────────────────────────
CONFIG_FILE = 'config.json'

SUPABASE_URL = "https://phonjftgqkutfeigdrts.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InBob25qZnRncWt1dGZlaWdkcnRzIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzE4Mjg5OTAsImV4cCI6MjA4NzQwNDk5MH0.w4ZHZEQXaYHCDMraFRsnRRM1WAfKRhXm25YwB6g33XM"

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ─── Logging ─────────────────────────────────────────────
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# ─── Supabase Helpers ────────────────────────────────────
def get_user_by_telegram(telegram_id: str):
    response = supabase.table("users").select("id, username").eq("telegram_id", telegram_id).execute()
    data = response.data
    return data[0] if data else None

def get_or_create_user(username: str, telegram_id: str):
    user = get_user_by_telegram(telegram_id)
    if user:
        if user.get("telegram_id") != telegram_id:
            supabase.table("users").update({"telegram_id": telegram_id}).eq("id", user["id"]).execute()
        return user["id"]

    data = {"username": username, "telegram_id": telegram_id}
    response = supabase.table("users").insert(data).execute()
    return response.data[0]["id"]

def get_session_by_telegram(telegram_id: str):
    user = get_user_by_telegram(telegram_id)
    if not user:
        user_id = get_or_create_user(f"telegram_user_{telegram_id}", telegram_id)
    else:
        user_id = user["id"]

    response = supabase.table("sessions") \
        .select("id") \
        .eq("user_id", user_id) \
        .order("created_at", desc=True) \
        .limit(1) \
        .execute()

    if response.data:
        return response.data[0]["id"]

    session_data = {
        "user_id": user_id,
        "title": f"Telegram Chat {datetime.now().strftime('%H:%M')}",
        "system_prompt": "You are a helpful assistant.",
        "persona_name": "Default Assistant"
    }
    resp = supabase.table("sessions").insert(session_data).execute()
    return resp.data[0]["id"]

def load_messages(session_id: int):
    response = supabase.table("messages") \
        .select("role, content") \
        .eq("session_id", session_id) \
        .order("timestamp", desc=False) \
        .execute()
    return [(m["role"], m["content"]) for m in response.data]

def save_message(session_id: int, role: str, content: str):
    data = {"session_id": session_id, "role": role, "content": content}
    supabase.table("messages").insert(data).execute()

def get_session_prompt(session_id: int):
    response = supabase.table("sessions").select("system_prompt").eq("id", session_id).execute()
    data = response.data
    return data[0]["system_prompt"] if data else "You are a helpful assistant."

def update_session_title(session_id: int, title: str):
    supabase.table("sessions").update({"title": title}).eq("id", session_id).execute()

def clear_session_messages(session_id: int):
    supabase.table("messages").delete().eq("session_id", session_id).execute()

def get_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f:
            return json.load(f)
    return {}

def simple_web_search(query: str, max_results=4):
    try:
        with DDGS() as ddgs:
            results = [r for r in ddgs.text(query, max_results=max_results)]
        return results
    except Exception as e:
        logger.error(f"Web search failed: {e}")
        return []

def send_to_nvidia(api_key, model, messages):
    url = "https://integrate.api.nvidia.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": 2048,
        "temperature": 0.7
    }
    response = requests.post(url, headers=headers, json=payload, timeout=120)
    response.raise_for_status()
    data = response.json()
    content = data['choices'][0]['message']['content']
    usage = data.get('usage', {})
    return content, usage

# ─── Bot Commands ────────────────────────────────────────
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    telegram_id = str(user.id)
    session_id = get_session_by_telegram(telegram_id)

    await update.message.reply_html(
        f"👋 Hi {user.mention_html()}!\n\n"
        f"🤖 NVIDIA AI Assistant\n\n"
        f"Commands:\n"
        f"/new — new session\n"
        f"/clear — clear current chat\n"
        f"/help — this message"
    )

async def new_session(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    telegram_id = str(user.id)
    user_id = get_or_create_user(f"telegram_user_{telegram_id}", telegram_id)

    session_data = {
        "user_id": user_id,
        "title": f"Chat {datetime.now().strftime('%H:%M')}",
        "system_prompt": "You are a helpful assistant.",
        "persona_name": "Default Assistant"
    }
    resp = supabase.table("sessions").insert(session_data).execute()
    session_id = resp.data[0]["id"]

    await update.message.reply_html(f"🆕 New session started!\nID: <code>{session_id}</code>")

async def clear_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    telegram_id = str(update.effective_user.id)
    session_id = get_session_by_telegram(telegram_id)
    clear_session_messages(session_id)
    await update.message.reply_text("🧹 Chat cleared. You can start fresh now!")

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_html(
        "🤖 Commands:\n"
        "/start — welcome\n"
        "/new — new conversation\n"
        "/clear — delete all messages in this chat\n"
        "/help — this message"
    )

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    telegram_id = str(user.id)
    text = update.message.text.strip()

    if not text:
        return

    config = get_config()
    api_key = config.get('api_key')
    model = config.get('model', 'meta/llama3-70b-instruct')

    if not api_key:
        await update.message.reply_text("❌ API key not set.")
        return

    session_id = get_session_by_telegram(telegram_id)
    system_prompt = get_session_prompt(session_id)

    await update.message.chat.send_action(action='typing')

    # Very basic trigger for web search
    search_context = ""
    search_triggers = ["current", "today", "news", "latest", "what's happening", "who won"]
    if any(t in text.lower() for t in search_triggers):
        results = simple_web_search(text, max_results=4)
        if results:
            search_context = "\n\nRecent web information:\n" + "\n".join(
                [f"• {r.get('title','')} : {r.get('body','')[:280]}" for r in results]
            )

    full_user_message = text + search_context
    save_message(session_id, 'user', full_user_message)

    messages = load_messages(session_id)
    api_messages = [{"role": "system", "content": system_prompt}]
    for role, content in messages:
        if role in ['user', 'assistant']:
            api_messages.append({"role": role, "content": content})

    try:
        response_text, usage = send_to_nvidia(api_key, model, api_messages)
        save_message(session_id, 'assistant', response_text)

        # Auto-title after 5–8 messages (only once-ish)
        msg_count = len(messages) + 1  # +1 for the new user message
        if 5 <= msg_count <= 8:
            try:
                title_prompt = [
                    {"role": "system", "content": "Create very short, descriptive title (4–8 words) for this conversation."},
                    {"role": "user", "content": "Summarize in title form:\n" + "\n".join([f"{r}: {c[:180]}" for r,c in messages[-6:]])}
                ]
                title_raw, _ = send_to_nvidia(api_key, model, title_prompt)
                title = title_raw.strip().strip('"').strip("'")[:60]
                if title and len(title) > 8:
                    update_session_title(session_id, title)
            except:
                pass  # silent fail

        # Prepare reply
        reply_text = response_text

        # Add usage info
        if usage:
            pt = usage.get('prompt_tokens', 0)
            ct = usage.get('completion_tokens', 0)
            total = pt + ct
            est_cost = round((total / 1_000_000) * 0.50, 5)  # very rough
            reply_text += f"\n\n<small>Tokens: {pt}+{ct} = {total}  •  ~${est_cost}</small>"

        if len(reply_text) > 4000:
            for chunk in [reply_text[i:i+4000] for i in range(0, len(reply_text), 4000)]:
                await update.message.reply_text(chunk, parse_mode='HTML')
        else:
            await update.message.reply_text(reply_text, parse_mode='HTML')

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        await update.message.reply_text(f"❌ Error: {str(e)}")
        save_message(session_id, 'system', f"Error: {str(e)}")

# ─── Main ────────────────────────────────────────────────
def main():
    config = get_config()
    token = config.get('telegram_token')
    if not token:
        logger.error("Telegram token missing")
        return

    logger.info("Starting bot...")

    application = Application.builder().token(token).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("new", new_session))
    application.add_handler(CommandHandler("clear", clear_command))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == '__main__':
    main()
