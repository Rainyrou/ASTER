# server_sandbox.py
import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "sandbox.server.server:app",
        host="0.0.0.0",
        port=8080,
        reload=False          # 开发时可改成 True
    )
