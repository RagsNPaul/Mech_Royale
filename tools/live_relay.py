"""Dev-only relay: play the public room with the local bot.

Not part of the match image. The websockets dependency lives in
requirements-dev.txt. The scored bot has no network.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import secrets
import subprocess
import sys
from pathlib import Path
from urllib.parse import quote

from websockets.asyncio.client import connect

_ROOT = Path(__file__).resolve().parents[1]
_HOST = "play.joinbuilders.org"


def _git_hash() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=_ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "dev"
    return out.strip() or "dev"


def _name(explicit: str | None) -> str:
    if explicit:
        return explicit
    return f"ragsbot-{_git_hash()}"


def _play_url(room: int, sid: str, name: str) -> str:
    return f"wss://{_HOST}/r/{room}/play?token=&sid={sid}&name={quote(name)}"


def _live_url(room: int, sid: str) -> str:
    return f"wss://{_HOST}/r/{room}/live?token=&sid={sid}"


class _Replay:
    """Live-socket lines for the match that is currently being played."""

    def __init__(self) -> None:
        self._file = None
        self._buf: list[str] = []

    def begin(self, path: Path) -> None:
        self.close()
        path.parent.mkdir(parents=True, exist_ok=True)
        self._file = path.open("w", encoding="utf-8")
        for line in self._buf:
            self._file.write(line + "\n")
        self._buf.clear()
        self._file.flush()

    def add(self, line: str) -> None:
        if self._file is None:
            self._buf.append(line)
            return
        self._file.write(line + "\n")
        self._file.flush()

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None


def _lines(raw: str | bytes) -> list[str]:
    text = raw.decode("utf-8", "replace") if isinstance(raw, bytes) else raw
    stripped = text.strip()
    if not stripped:
        return []
    if stripped.startswith("{") and "\n" not in stripped:
        return [stripped]
    return [line.strip() for line in text.splitlines() if line.strip()]


async def _read_live(ws, replay: _Replay) -> None:
    async for raw in ws:
        for line in _lines(raw):
            replay.add(line)


async def _pump_stderr(proc: asyncio.subprocess.Process, path: Path) -> None:
    assert proc.stderr is not None
    with path.open("w", encoding="utf-8") as handle:
        while True:
            line = await proc.stderr.readline()
            if not line:
                return
            handle.write(line.decode("utf-8", "replace"))
            handle.flush()


async def _pump_stdout(proc: asyncio.subprocess.Process, actions: Path, play) -> None:
    assert proc.stdout is not None
    with actions.open("w", encoding="utf-8") as handle:
        while True:
            line = await proc.stdout.readline()
            if not line:
                return
            text = line.decode("utf-8", "replace").strip()
            if not text:
                continue
            try:
                msg = json.loads(text)
            except json.JSONDecodeError:
                continue
            if isinstance(msg, dict) and msg.get("type") == "ready":
                continue
            handle.write(text + "\n")
            handle.flush()
            try:
                await play.send(text)
            except Exception as exc:
                print(f"relay: action send failed: {exc}", file=sys.stderr)
                return


async def _finish_pumps(tasks: list[asyncio.Task]) -> None:
    if not tasks:
        return
    try:
        await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), 1)
    except TimeoutError:
        pass


async def _stop(proc: asyncio.subprocess.Process | None, tasks: list[asyncio.Task]) -> None:
    for task in tasks:
        task.cancel()
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)
    if proc is None:
        return
    if proc.returncode is None:
        proc.kill()
        try:
            await proc.wait()
        except ProcessLookupError:
            pass


async def _to_bot(proc: asyncio.subprocess.Process, payload: bytes) -> None:
    assert proc.stdin is not None
    if proc.stdin.is_closing():
        return
    try:
        proc.stdin.write(payload)
        await proc.stdin.drain()
    except (BrokenPipeError, ConnectionResetError, RuntimeError):
        pass


def _play_messages(raw: str | bytes) -> list[dict]:
    text = raw.decode("utf-8", "replace") if isinstance(raw, bytes) else raw
    if not text.strip():
        return []
    try:
        msg = json.loads(text)
    except json.JSONDecodeError:
        msg = None
    if isinstance(msg, dict):
        return [msg]
    out = []
    for line in _lines(text):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            out.append(row)
    return out


async def _read_play(play, queue: asyncio.Queue) -> None:
    try:
        async for raw in play:
            for msg in _play_messages(raw):
                await queue.put(msg)
    finally:
        await queue.put(None)


async def _play_match(play, hello: dict, cmd: list[str], directory: Path, replay: _Replay, queue: asyncio.Queue) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    replay.begin(directory / "replay.jsonl")
    hello = json.loads(json.dumps(hello))
    mp = hello.get("map")
    if not isinstance(mp, dict):
        mp = {}
        hello["map"] = mp
    mp["dir"] = str((_ROOT / "arena").resolve())
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=_ROOT,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    assert proc.stdin is not None
    tasks = [
        asyncio.create_task(_pump_stdout(proc, directory / "actions.jsonl", play)),
        asyncio.create_task(_pump_stderr(proc, directory / "bot_stderr.txt")),
    ]
    obs_path = directory / "obs.jsonl"
    try:
        await _to_bot(proc, (json.dumps(hello, separators=(",", ":")) + "\n").encode())
        with obs_path.open("w", encoding="utf-8") as obs_file:
            while True:
                msg = await queue.get()
                if msg is None:
                    return
                kind = msg.get("type")
                if kind == "obs":
                    encoded = json.dumps(msg, separators=(",", ":"))
                    obs_file.write(encoded + "\n")
                    obs_file.flush()
                    await _to_bot(proc, (encoded + "\n").encode())
                elif kind == "end":
                    (directory / "end.json").write_text(json.dumps(msg, indent=2) + "\n", encoding="utf-8")
                    await _to_bot(proc, (json.dumps(msg, separators=(",", ":")) + "\n").encode())
                    try:
                        proc.stdin.close()
                    except (BrokenPipeError, ConnectionResetError, RuntimeError):
                        pass
                    try:
                        await asyncio.wait_for(proc.wait(), 3)
                    except TimeoutError:
                        proc.kill()
                        await proc.wait()
                    await _finish_pumps(tasks)
                    return
                elif kind == "hello":
                    raise _NextHello(msg)
    finally:
        await _stop(proc, tasks)


class _NextHello(Exception):
    def __init__(self, hello: dict) -> None:
        self.hello = hello


async def _session(args: argparse.Namespace) -> None:
    sid = secrets.token_hex(16)
    name = _name(args.name)
    room = int(args.room)
    out = Path(args.out)
    if not out.is_absolute():
        out = _ROOT / out
    cmd = args.cmd or ["python3", "-u", "main.py"]
    print(f"relay: room {room} name {name!r} sid {sid} matches {args.matches}", file=sys.stderr)
    replay = _Replay()
    queue: asyncio.Queue = asyncio.Queue()
    async with connect(_play_url(room, sid, name), max_size=None) as play, connect(
        _live_url(room, sid), max_size=None
    ) as live:
        live_task = asyncio.create_task(_read_live(live, replay))
        play_task = asyncio.create_task(_read_play(play, queue))
        try:
            await play.send(json.dumps({"type": "ready", "name": name}))
            pending: dict | None = None
            played = 0
            while played < int(args.matches):
                hello = pending
                pending = None
                if hello is None:
                    hello = await _next_hello(queue)
                    if hello is None:
                        break
                played += 1
                print(f"relay: match {played} slot {hello.get('slot')} seed {hello.get('seed')}", file=sys.stderr)
                try:
                    await _play_match(play, hello, cmd, out / str(played), replay, queue)
                except _NextHello as nxt:
                    pending = nxt.hello
        finally:
            play_task.cancel()
            live_task.cancel()
            replay.close()
            for task in (play_task, live_task):
                try:
                    await task
                except asyncio.CancelledError:
                    pass
    print("relay: disconnected", file=sys.stderr)


async def _next_hello(queue: asyncio.Queue) -> dict | None:
    while True:
        msg = await queue.get()
        if msg is None:
            return None
        if msg.get("type") == "hello":
            return msg


def main() -> None:
    parser = argparse.ArgumentParser(description="Play the public mechfight room with the local bot.")
    parser.add_argument("--room", type=int, default=3)
    parser.add_argument("--matches", type=int, default=5)
    parser.add_argument("--name", default=None, help='default "ragsbot-<git short hash>"')
    parser.add_argument("--out", default="runs")
    parser.add_argument("--cmd", nargs=argparse.REMAINDER, default=None, help="bot command (default: python3 -u main.py)")
    args = parser.parse_args()
    if args.cmd and args.cmd[0] == "--":
        args.cmd = args.cmd[1:]
    try:
        asyncio.run(_session(args))
    except KeyboardInterrupt:
        print("relay: interrupted", file=sys.stderr)


if __name__ == "__main__":
    main()
