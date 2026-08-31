"""iPad preference slider: one continuous Efficiency <-> Push Intensity choice.

Run this on the external computer, then open the printed URL in Safari on the
iPad. The slider is always live: drag it anywhere, hit Send, and the index it
sits at is turned into a 3-D action and handed to the device.

    pip install websockets

Usage:

    from preference import Preference

    p = Preference()
    p.wait_for_ipad()
    p.serve_forever()      # each Send -> Exo.send(action_for(idx))

The index is 0 - 100, 0 at the Efficiency end and 100 at the Push Intensity end, and
`action_for` blends the two endpoint actions by it.

Two servers run in daemon threads: HTTP serves this directory, and a websocket
carries `slider`/`submit` back. The ports differ from `survey.py`'s so both can
run at once.
"""

import asyncio
import functools
import http.server
import json
import queue
import threading
from pathlib import Path

import numpy as np
import websockets

import pypolar as plr
from tablet.survey import local_ips

HTTP_PORT = 8001
WS_PORT   = 8766

PAGE = 'preference.html'

LABELS = ('Efficiency', 'Push Intensity')      # the slider's low and high ends

EFF_ACTION = np.array([-2.0,  1.0, 0.5])    # action at index 0
COM_ACTION = np.array([ 3.0, -1.5, 2.0])    # action at index 100


def action_for(idx) -> np.ndarray:
    """The 3-D action at slider index `idx`, 0 - 100, as a blend of the ends."""
    w = float(idx) / 100.0
    return (1.0 - w) * EFF_ACTION + w * COM_ACTION


class Exo(plr.Device):

    def send(self, action):
        print('Exo got action', action)


class _PageHandler(http.server.SimpleHTTPRequestHandler):
    """Serves this directory, sending the bare root to `PAGE`."""

    quiet = False       # set per Preference, on a subclass made in __init__

    def do_GET(self):
        if self.path == '/':
            self.send_response(302)
            self.send_header('Location', '/' + PAGE)
            self.end_headers()
            return

        super().do_GET()

    def log_message(self, fmt, *args):
        if not self.quiet:
            super().log_message(fmt, *args)


class Preference:
    """A continuous Efficiency <-> Push Intensity slider on an iPad.

    Args:
        device: what a submitted action is sent to.
        labels: the two end names the page shows, low end first.
        http_port, ws_port: ports the two servers listen on.
        quiet: silences the HTTP request log.
    """

    def __init__(self, device=None, labels=LABELS, http_port=HTTP_PORT,
                 ws_port=WS_PORT, quiet=True):
        # set before the servers start, since a client may connect immediately
        self.device = Exo() if device is None else device
        self.labels = tuple(labels)
        self.slider = 50        # where the slider is right now, 0 - 100

        self._sends     = queue.Queue()
        self._ws_port   = ws_port
        self._clients   = set()
        self._loop      = None
        self._stop      = None
        self._connected = threading.Event()
        self._ready     = threading.Event()

        handler = functools.partial(
            type('_Handler', (_PageHandler,), {'quiet': quiet}),
            directory=str(Path(__file__).resolve().parent),
        )
        self._http = http.server.ThreadingHTTPServer(('', http_port), handler)
        threading.Thread(target=self._http.serve_forever, daemon=True).start()

        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        self._ready.wait(5)

        ips = local_ips()
        if len(ips) == 1:
            print(f'Open this on the iPad:  http://{ips[0]}:{http_port}/{PAGE}')
        else:
            print('Open one of these on the iPad (whichever shares its Wi-Fi):')
            for ip in ips:
                print(f'    http://{ip}:{http_port}/{PAGE}')

    # ---- sends ---------------------------------------------------------------

    def wait_for_send(self, timeout=None):
        """Block until Send is tapped and return the index (None on timeout)."""
        try:
            return self._sends.get(timeout=timeout)
        except queue.Empty:
            return None

    def serve_forever(self):
        """Send every submitted index's action to the device, until interrupted."""
        while True:
            idx = self.wait_for_send()
            if idx is None:
                continue
            self.device.send(action_for(idx))

    def clear_sends(self):
        """Throw away any taps that have queued up but not been read."""
        while not self._sends.empty():
            try:
                self._sends.get_nowait()
            except queue.Empty:
                break

    def wait_for_ipad(self, timeout=None):
        """Block until the page connects. Returns True if it did."""
        return self._connected.wait(timeout)

    @property
    def connected(self):
        return bool(self._clients)

    def close(self):
        """Shut both servers down cleanly."""
        self._http.shutdown()
        if self._loop and self._stop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._stop.set)
            self._thread.join(timeout=3)

    # ---- plumbing ------------------------------------------------------------

    async def _handler(self, ws, *_):
        self._clients.add(ws)
        self._connected.set()
        print('iPad connected')
        # the page ships with the default ends; name them for this run
        await ws.send(json.dumps({'labels': list(self.labels)}))
        try:
            async for raw in ws:
                try:
                    msg = json.loads(raw)
                except ValueError:
                    continue
                if 'slider' in msg:
                    self.slider = int(msg['slider'])

                if 'submit' in msg:
                    self.slider = int(msg['submit'])
                    self._sends.put(self.slider)
        except Exception:
            pass
        finally:
            self._clients.discard(ws)
            if not self._clients:
                self._connected.clear()
            print('iPad disconnected')

    def _run_loop(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        self._stop = asyncio.Event()

        async def main():
            async with websockets.serve(self._handler, '', self._ws_port):
                self._ready.set()
                await self._stop.wait()
                for ws in list(self._clients):
                    await ws.close()

        self._loop.run_until_complete(main())
        self._loop.close()


if __name__ == '__main__':
    p = Preference()
    print('Waiting for the iPad...')
    p.wait_for_ipad()

    try:
        p.serve_forever()
    except KeyboardInterrupt:
        p.close()
