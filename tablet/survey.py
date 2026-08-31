"""iPad push-intensity survey: a 1 - 5 scale that opens one question at a time.

Run this on the external computer, then open the printed URL in Safari on the
iPad. The page is greyed out and untouchable until `ask` opens a question; from
then the subject has `timeout` seconds to pick a position and hit Submit.

    pip install websockets

Usage:

    from survey import Survey

    s = Survey()
    s.wait_for_ipad()

    s.ask()               # (1,) rating, or (0,) if the window closed unanswered
    s.close()

`ask` is what a `pypolar.Probe` calls, from the probe's own worker thread. The
empty array is how a missed question reaches the experiment: `Probe` still counts
the repeat, `Probe.data` drops it, and `Logger.end_trial` adds nothing to the
objective for a trial whose every repeat timed out.

Two servers run in daemon threads: HTTP serves this directory, and a websocket
carries `arm`/`disarm`/`hold` out and `slider`/`submit` back.
"""

import asyncio
import functools
import http.server
import json
import queue
import socket
import subprocess
import threading
import time
import logging
from pathlib import Path

import numpy as np
import websockets

HTTP_PORT = 8000
WS_PORT   = 8765

PAGE    = 'survey.html'
TIMEOUT = 25.0                # seconds the subject has to answer
PERIOD  = 30.0                # wall clock one question occupies, answered early or not


def _route_ip():
    """Source address the OS would route to the internet (no traffic is sent)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('192.0.2.1', 1))
        return s.getsockname()[0]
    except OSError:
        return '127.0.0.1'
    finally:
        s.close()


def _lan_rank(ip):
    """Sort key preferring the address ranges a home/office Wi-Fi actually uses."""
    for i, prefix in enumerate(('192.168.', '172.', '10.')):
        if ip.startswith(prefix):
            return i
    return 3


def local_ips():
    """This machine's LAN IPv4 addresses, likeliest first, VPN tunnels excluded.

    The route trick alone returns the VPN's tunnel address when a VPN is up, and
    the iPad cannot reach that; so enumerate every interface instead and drop the
    point-to-point (VPN) ones. Which of the remaining is the right Wi-Fi is not
    always decidable here (a VM bridge looks like a LAN), so all are reported.
    """
    try:
        out = subprocess.check_output(['ifconfig'], text=True)
    except (OSError, subprocess.CalledProcessError):
        return [_route_ip()]

    addrs = []
    for line in out.splitlines():
        line = line.strip()
        if not line.startswith('inet ') or '-->' in line:   # skip IPv6 and point-to-point (VPN) links
            continue
        ip = line.split()[1]
        if ip == '127.0.0.1' or ip.startswith('169.254.'):
            continue
        addrs.append(ip)

    addrs.sort(key=_lan_rank)
    route = _route_ip()                                     # promote the routed one if it isn't a tunnel
    if route in addrs:
        addrs.insert(0, addrs.pop(addrs.index(route)))
    return addrs or [route]


class _PageHandler(http.server.SimpleHTTPRequestHandler):
    """Serves this directory, sending the bare root to `PAGE`.

    Without the redirect a mistyped or bookmarked root URL 404s, since the page
    is not named index.html.
    """

    quiet = False       # set per Survey, on a subclass made in __init__

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

# kept local rather than imported: this module is standalone. The key is the
# contract with hilo.log.ConsoleFilter.
TO_BOTH = {"console": True}

class Survey:
    """A rating scale on an iPad, armed one question at a time.

    Args:
        timeout: seconds a question stays open.
        period: seconds the whole question occupies, so a subject who answers
            early waits out the rest before the next one opens. A period below
            the timeout just means no wait, never a shortened question.
        http_port, ws_port: ports the two servers listen on.
        quiet: silences the HTTP request log.
    """

    def __init__(self, logger: logging.Logger, timeout=TIMEOUT, period=PERIOD, http_port=HTTP_PORT,
                 ws_port=WS_PORT, quiet=True):
        # set before the servers start, since a client may connect immediately
        self.timeout   = float(timeout)
        self.period    = float(period)
        self.slider    = 3         # where the slider is right now, 1 - 5
        self.trial     = None      # trial the caller last named
        self.repeat    = 0         # questions asked so far within it, 1-based
        self._deadline = None

        self._sends     = queue.Queue()
        self._ws_port   = ws_port
        self._clients   = set()
        self._loop      = None
        self._stop      = None
        self._connected = threading.Event()
        self._ready     = threading.Event()
        self.logger = logger

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
            logger.info(f'Open this on the iPad:  http://{ips[0]}:{http_port}/{PAGE}', extra=TO_BOTH)
        else:
            logger.info('Open one of these on the iPad (whichever shares its Wi-Fi):', extra=TO_BOTH)
            for ip in ips:
                logger.info(f'    http://{ip}:{http_port}/{PAGE}', extra=TO_BOTH)

    # ---- one question --------------------------------------------------------

    def ask(self, action=None, trial=None, timeout=None, period=None) -> np.ndarray:
        """Opens one question and blocks out the full period.

        Args:
            action: what the subject is rating. Unused here; the probe passes it.
            trial: which trial this question belongs to. A `Probe` calls this
                once per repeat with the arguments it was given, so a repeated
                trial is how the repeats within one action are counted.
            timeout: seconds to wait, defaulting to the survey's own.
            period: seconds the call takes in total, defaulting to the survey's.

        Returns:
            (1,) the rating, or (0,) when the window closed unanswered.
        """
        timeout = self.timeout if timeout is None else float(timeout)
        period  = self.period  if period  is None else float(period)
        start   = time.monotonic()

        if trial is not None and trial != self.trial:
            self.trial  = trial
            self.repeat = 0
        self.repeat += 1

        # a tap that landed after the last window closed is not this answer
        self.clear_sends()
        self.arm(timeout, trial=self.trial, repeat=self.repeat)
        try:
            value = self.wait_for_send(timeout=timeout, discard_stale=False)
        finally:
            self.disarm()

        self.hold(period - (time.monotonic() - start))
        return np.empty(0) if value is None else np.atleast_1d(float(value))

    def hold(self, seconds):
        """Blocks for the rest of the period, counting it down on the grey page."""
        if seconds <= 0:
            return

        self._send({'hold': seconds})
        time.sleep(seconds)

    def arm(self, seconds=None, trial=None, repeat=None):
        """Makes the scale interactable and starts the countdown.

        `trial` and `repeat` are labels for the page: a repeat above the first
        is the same action rated again, which the page says out loud.
        """
        seconds = self.timeout if seconds is None else float(seconds)
        self._deadline = time.monotonic() + seconds
        self._send({'arm': seconds, 'trial': trial, 'repeat': repeat})

    def disarm(self):
        """Greys the scale out again."""
        self._deadline = None
        self._send({'disarm': True})

    @property
    def seconds_left(self) -> float:
        """Seconds left on the open question, 0.0 when none is open."""
        if self._deadline is None:
            return 0.0

        return max(0.0, self._deadline - time.monotonic())

    # ---- answers and connection ----------------------------------------------

    def wait_for_send(self, timeout=None, discard_stale=True):
        """Block until Submit is tapped and return the rating (None on timeout).

        discard_stale drops any taps that arrived before this call, so a
        double-tap on the previous question can't satisfy the next one.
        """
        if discard_stale:
            self.clear_sends()
        try:
            return self._sends.get(timeout=timeout)
        except queue.Empty:
            return None

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

    def _send(self, msg):
        if self._loop is None:
            return
        asyncio.run_coroutine_threadsafe(self._broadcast(json.dumps(msg)), self._loop)

    async def _broadcast(self, text):
        for ws in list(self._clients):
            try:
                await ws.send(text)
            except Exception:
                self._clients.discard(ws)

    async def _handler(self, ws, *_):
        self._clients.add(ws)
        self._connected.set()
        self.logger.debug('iPad connected')
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
            self.logger.debug('iPad disconnected')

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
    s = Survey()
    print('Waiting for the iPad...')
    s.wait_for_ipad()

    # two repeats per trial, as a Probe with repeats=2 would ask them
    try:
        trial = 0
        while True:
            trial += 1
            for _ in range(2):
                print(f'Asking, trial {trial}...')
                value = s.ask(trial=trial)
                print('got', value if len(value) else 'nothing (timed out)')
    except KeyboardInterrupt:
        s.close()
