# SETUP.md

## Code Copilot Server + Ollama Setup Guide

### 1. **Install Prerequisites**

- **Python 3.8+** (recommended: 3.10+)
- **Ollama** (for local LLM inference):  
  [Install Ollama](https://ollama.com/download) for your OS.
- **Git** (optional, for codebase management)

---

### 2. **Clone the Repository**

```bash
git clone <your-repo-url>
cd code_copilot
```

---

### 3. **Set Up Python Environment**

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

### 4. **Start Ollama**

Pull and run your desired model (e.g., Qwen2.5, Llama3, etc.):

```bash
ollama pull qwen2.5
ollama run qwen2.5
```
*(Leave this terminal running, or run Ollama as a background service.)*

---

### 5. **Configure the Server**

Check or edit `server_config.json` to ensure it points to your server script:

```json
{
  "mcpServers": {
    "code-copilot": {
      "command": "python",
      "args": ["fastmcp_server.py"],
      "env": {}
    }
  }
}
```

---

### 6. **Start the Code Copilot Server**

```bash
python fastmcp_server.py
```

---

### 7. **Start the MCP Client**

In a new terminal (with the virtual environment activated):

```bash
python mcp_client.py
```

---

### 8. **Interact with the System**

- Type your questions or requests in the chat.
- Use `tools` to list available tools.
- Use `resources` to list available resources.
- The LLM (via Ollama) will call tools on the server as needed.

---

### 9. **Troubleshooting**

- **Ollama not running:**  
  Make sure `ollama run <model>` is active.
- **Module not found:**  
  Run `pip install -r requirements.txt` in your venv.
- **Server not starting:**  
  Remove or fix any `@app.get_resources` decorators (use only `@app.resource` and `@app.tool`).

---

### 10. **(Optional) Customization**

- Change the Ollama model in `mcp_client.py` if desired.
- Add new tools or prompts in `fastmcp_server.py`.
- Integrate with a web or desktop UI if needed.

---

**You are now ready to use your own private, cost-effective, and extensible AI code copilot!**
