"""FastAPI endpoints preserving the shared TypeScript wire contracts."""

import asyncio
import json
import os
import signal
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Timer

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse

from .cancellation import ScanCancelled
from .config import choose_library, read_config
from .indexer import initial_progress
from .service import Service

DIST = Path(__file__).resolve().parents[2] / "dist"
SHUTDOWN_TIMEOUT = 5


class GalleryServer(uvicorn.Server):
    """Notify the app before Uvicorn waits for long-lived HTTP streams."""

    def handle_exit(self, sig, frame):
        if self.should_exit and sig == signal.SIGINT:
            # Replaying repeated SIGINT through asyncio cancels lifespan cleanup
            # and leaves Python joining a blocked inference thread at exit.
            os._exit(130)
        # Uvicorn still installs/restores the signal handlers. Locallery consumes
        # the signal here rather than recording it for replay after cleanup.
        self.should_exit = True

    async def shutdown(self, sockets=None):
        app = self.config.loaded_app
        while app is not None:
            callback = getattr(getattr(app, "state", None), "begin_shutdown", None)
            if callback is not None:
                callback()
                break
            app = getattr(app, "app", None)
        await super().shutdown(sockets)


def create_app(config=None, embedder_factory=None):
    config = config or read_config()
    executor = ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="locallery-inference"
    )
    subscribers = set()
    state = {
        "progress": initial_progress(),
        "pending": 0,
        "last_log": 0,
        "last_publish": 0,
        "last_stage": "",
    }
    loop = None
    scan_task = None

    def publish(progress):
        state["progress"] = progress
        now = time.monotonic()
        if (
            not progress["busy"]
            or progress["stage"] != state["last_stage"]
            or now - state["last_publish"] >= 0.1
        ):
            state["last_publish"], state["last_stage"] = now, progress["stage"]
            for queue in subscribers:
                if queue.full():
                    queue.get_nowait()
                queue.put_nowait(progress)
        if now - state["last_log"] >= 1 or not progress["busy"]:
            state["last_log"] = now
            eta = (
                ""
                if progress["etaSeconds"] is None
                else f" · ETA {progress['etaSeconds']:.0f}s"
            )
            print(
                f"[{progress['stage']}] {progress['message']} | {progress['processed']}/{progress['total']} · indexed {progress['indexed']} · unchanged {progress['unchanged']} · skipped {progress['skipped']} · errors {progress['failed']} · {progress['rate']:.1f}/s{eta}",
                flush=True,
            )

    def report(progress):
        loop.call_soon_threadsafe(publish, progress)

    kwargs = {"embedder_factory": embedder_factory} if embedder_factory else {}
    service = Service(config, report, **kwargs)
    shutdown_timer = None

    def begin_shutdown():
        nonlocal shutdown_timer
        if service.stopping.is_set():
            return
        service.stop()
        for queue in subscribers:
            if queue.full():
                queue.get_nowait()
            queue.put_nowait(None)

        def force_shutdown():
            # Python cannot safely interrupt native inference on another thread.
            # Completed SQLite writes are already committed; restart rescans.
            print(
                f"Active operation did not stop within {SHUTDOWN_TIMEOUT:g} seconds; exiting.",
                flush=True,
            )
            os._exit(130)

        shutdown_timer = Timer(SHUTDOWN_TIMEOUT, force_shutdown)
        shutdown_timer.daemon = True
        shutdown_timer.start()

    async def call(function, *args):
        return await asyncio.get_running_loop().run_in_executor(
            executor, function, *args
        )

    @asynccontextmanager
    async def lifespan(app):
        nonlocal loop, scan_task
        loop = asyncio.get_running_loop()
        scan_task = asyncio.create_task(call(service.rescan))
        print(
            f"Locallery → http://{config.host}:{config.port} · Transformers → {config.model}",
            flush=True,
        )
        try:
            yield
        finally:
            begin_shutdown()
            cleanup_complete = False
            try:
                if scan_task:
                    await scan_task
                await call(service.close)
                executor.shutdown(wait=True)
                cleanup_complete = True
            finally:
                if shutdown_timer and cleanup_complete:
                    shutdown_timer.cancel()

    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    app.state.gallery = state
    app.state.begin_shutdown = begin_shutdown

    @app.middleware("http")
    async def indexing_gate(request, next_handler):
        if service.stopping.is_set():
            return JSONResponse({"error": "Server is shutting down"}, status_code=503)
        if (
            request.url.path.startswith("/api/")
            and request.url.path not in ("/api/status", "/api/events", "/api/rescan")
            and state["progress"]["busy"]
        ):
            return JSONResponse({"error": "Indexing is underway"}, status_code=503)
        return await next_handler(request)

    @app.exception_handler(ValueError)
    async def invalid(request, exception):
        return JSONResponse(
            {"error": str(exception)},
            status_code=404 if "not found" in str(exception).lower() else 400,
        )

    @app.exception_handler(ScanCancelled)
    async def stopped(request, exception):
        return JSONResponse({"error": "Server is shutting down"}, status_code=503)

    @app.exception_handler(Exception)
    async def failed(request, exception):
        print(f"Request failed: {exception}", flush=True)
        return JSONResponse({"error": str(exception)}, status_code=500)

    @app.get("/api/status")
    async def status():
        return state["progress"]

    @app.get("/api/events")
    async def events(request: Request):
        async def stream():
            queue = asyncio.Queue(maxsize=2)
            subscribers.add(queue)
            try:
                yield f"data: {json.dumps(state['progress'])}\n\n"
                while (
                    not service.stopping.is_set()
                    and not await request.is_disconnected()
                ):
                    try:
                        progress = await asyncio.wait_for(queue.get(), timeout=15)
                        if progress is None:
                            break
                        yield f"data: {json.dumps(progress)}\n\n"
                    except TimeoutError:
                        yield ": keepalive\n\n"
            finally:
                subscribers.discard(queue)

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"cache-control": "no-cache", "x-accel-buffering": "no"},
        )

    @app.post("/api/rescan")
    async def rescan():
        nonlocal scan_task
        if (
            state["progress"]["busy"]
            or state["pending"]
            or (scan_task and not scan_task.done())
        ):
            return JSONResponse({"error": "A job is already running"}, status_code=409)
        publish(initial_progress())
        scan_task = asyncio.create_task(call(service.rescan))
        return JSONResponse({"accepted": True}, status_code=202)

    @app.get("/api/folders")
    async def get_folders():
        return await call(service.get_folders)

    @app.get("/api/images")
    async def get_images(request: Request):
        try:
            page, size = (
                int(request.query_params.get("page", "1")),
                int(request.query_params.get("pageSize", "96")),
            )
            group = request.query_params.get("groupId")
            group = int(group) if group is not None else None
        except ValueError:
            raise ValueError("Invalid pagination or group") from None
        if page < 1 or not 1 <= size <= 200 or (group is not None and group < 0):
            raise ValueError("Invalid pagination or group")
        return await call(
            service.get_images, page, size, request.query_params.get("folderId"), group
        )

    @app.get("/api/groups")
    async def get_groups():
        return await call(service.get_groups)

    @app.post("/api/search")
    async def search(request: Request):
        data = bytearray()
        async for chunk in request.stream():
            data.extend(chunk)
            if len(data) > 10000:
                return JSONResponse({"error": "Request too large"}, status_code=413)
        try:
            body = json.loads(data)
        except (ValueError, UnicodeDecodeError):
            raise ValueError("Invalid JSON") from None
        if not isinstance(body, dict):
            raise ValueError("Search must be an object")
        clean = {}
        for key in ("query", "folderId", "referenceImageId"):
            if key in body:
                if not isinstance(body[key], str) or len(body[key]) > 2000:
                    raise ValueError(f"Invalid {key}")
                clean[key] = body[key]
        state["pending"] += 1
        try:
            return await call(service.search, clean)
        finally:
            state["pending"] -= 1

    @app.get("/api/images/{image_id}")
    async def image(image_id: str):
        return await call(service.get_image, image_id)

    @app.get("/api/images/{image_id}/{kind}")
    async def image_file(image_id: str, kind: str):
        if kind not in ("preview", "original"):
            return JSONResponse({"error": "Not found"}, status_code=404)
        file = await call(service.image_file, image_id, kind)
        return FileResponse(
            file,
            media_type="image/jpeg" if kind == "preview" else None,
            headers={"cache-control": "no-cache", "x-content-type-options": "nosniff"},
        )

    @app.api_route("/{path:path}", methods=["GET", "HEAD"])
    async def frontend(path: str):
        if path.startswith("api/"):
            return JSONResponse({"error": "Not found"}, status_code=404)
        file = (DIST / path).resolve()
        if not file.is_relative_to(DIST):
            return Response("Forbidden", status_code=403)
        if not file.is_file():
            file = DIST / "index.html"
        if not file.exists():
            return Response("Frontend not built. Run bun run build.", status_code=503)
        return FileResponse(file)

    return app


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Locallery local gallery")
    parser.add_argument(
        "--reload", action="store_true", help="Reload Python source during development"
    )
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="Print server bind settings for the Vite launcher",
    )
    parser.add_argument(
        "--library", help="Use this album folder without the interactive prompt"
    )
    parser.add_argument("--select-library", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not args.print_config or args.select_library:
        try:
            library = choose_library(args.library)
        except ValueError as error:
            parser.error(str(error))
        # Reload workers inherit the choice, including transient CLI overrides.
        if library is None:
            os.environ.pop("LOCALLERY_LIBRARY", None)
        else:
            os.environ["LOCALLERY_LIBRARY"] = str(library)
    config = read_config()
    if args.print_config:
        print(
            json.dumps(
                {
                    "host": config.host,
                    "port": config.port,
                    "library": str(config.library),
                }
            )
        )
        return
    server_config = uvicorn.Config(
        "locallery.server:create_app" if args.reload else create_app(config),
        factory=args.reload,
        reload=args.reload,
        reload_dirs=[str(Path(__file__).parent)] if args.reload else None,
        host=config.host,
        port=config.port,
        timeout_keep_alive=255,
        timeout_graceful_shutdown=5,
    )
    server = GalleryServer(server_config)
    try:
        if server_config.should_reload:
            from uvicorn.supervisors import ChangeReload

            socket = server_config.bind_socket()
            ChangeReload(server_config, target=server.run, sockets=[socket]).run()
        else:
            server.run()
    except KeyboardInterrupt:
        pass
    if not server.started and not server_config.should_reload:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
