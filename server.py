from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI()
trigger_message = None

@app.post("/trigger")
async def set_trigger(message: str):
    global trigger_message
    trigger_message = message
    return {"status": "trigger set"}

@app.get("/check")
async def check_trigger():
    global trigger_message
    if trigger_message:
        msg = trigger_message
        trigger_message = None  # очищаем триггер после отработки
        return JSONResponse(content=msg)
    else:
        return JSONResponse(content="")