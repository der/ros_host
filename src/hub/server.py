"""Hub ASGI server - central message broker using python-socketio."""

import logging

import socketio
import uvicorn
from starlette.applications import Starlette
from starlette.routing import Mount

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger("hub")

sio = socketio.AsyncServer(async_mode="asgi", cors_allowed_origins="*")
sio_app = socketio.ASGIApp(sio)

rooms: dict[str, set[str]] = {}


@sio.event
async def connect(sid, environ):
    logger.info(f"Client connected: {sid}")


@sio.event
async def disconnect(sid):
    logger.info(f"Client disconnected: {sid}")
    for room, members in list(rooms.items()):
        members.discard(sid)
        if not members:
            del rooms[room]


@sio.event
async def join(sid, data):
    room = data.get("room")
    if not room:
        logger.warning(f"Client {sid} sent join without room")
        return
    await sio.enter_room(sid, room)
    rooms.setdefault(room, set()).add(sid)
    logger.info(f"Client {sid} joined room: {room}")


@sio.event
async def leave(sid, data):
    room = data.get("room")
    if not room:
        return
    await sio.leave_room(sid, room)
    rooms.get(room, set()).discard(sid)
    logger.info(f"Client {sid} left room: {room}")


@sio.event
async def publish(sid, data):
    room = data.get("room")
    if not room:
        logger.warning(f"Client {sid} sent publish without room")
        return

    message = data.get("message")

    # logger.info(f"Client {sid} published to room: {room}")

    await sio.emit(
        "message",
        {"room": room, "message": message},
        room=room,
        # skip_sid=sid,
    )


app = Starlette(
    routes=[
        Mount("/", app=sio_app),
    ]
)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)

