import threading, time
import webview
from app import app

PORT = 5317

def run_server():
    app.run(host="127.0.0.1", port=PORT, debug=False, use_reloader=False)

if __name__ == "__main__":
    threading.Thread(target=run_server, daemon=True).start()
    time.sleep(0.8)
    webview.create_window("FSL Learning", f"http://127.0.0.1:{PORT}", width=1200, height=800)
    webview.start()
