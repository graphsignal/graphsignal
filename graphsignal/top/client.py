"""HTTP client for the watcher's `/signals` endpoint."""

import json
import os
import socket
import urllib.error
import urllib.request


class SignalsUnavailable(Exception):
    """The endpoint could not be read; `str(exc)` is a short reason for the header."""


_REPLACEMENT = '\ufffd'


def printable(value):
    """`value` with every string in it made safe to write to a terminal:
    characters that are not printable become U+FFFD, newlines and tabs stay."""
    if isinstance(value, str):
        if value.isprintable():
            return value
        return ''.join(
            ch if ch.isprintable() or ch in '\n\t' else _REPLACEMENT for ch in value)
    if isinstance(value, list):
        return [printable(item) for item in value]
    if isinstance(value, dict):
        return {printable(key): printable(item) for key, item in value.items()}
    return value


class SignalsClient:
    def __init__(self, host, port, timeout=2.0, token=None):
        self.host = str(host)
        self.port = int(port)
        self.timeout = float(timeout)
        self.token = token if token is not None else os.getenv('GRAPHSIGNAL_LISTEN_TOKEN')

    @property
    def endpoint(self):
        host = f'[{self.host}]' if ':' in self.host else self.host
        return f'{host}:{self.port}'

    @property
    def url(self):
        return f'http://{self.endpoint}/signals'

    def fetch(self):
        """The parsed `/signals` payload, or raise `SignalsUnavailable`."""
        headers = {'Accept': 'application/json'}
        if self.token:
            headers['Authorization'] = f'Bearer {self.token}'
        request = urllib.request.Request(self.url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                if response.status != 200:
                    raise SignalsUnavailable(f'HTTP {response.status}')
                body = response.read()
        except urllib.error.HTTPError as err:
            raise SignalsUnavailable(f'HTTP {err.code}') from err
        except urllib.error.URLError as err:
            reason = err.reason
            if isinstance(reason, socket.timeout):
                raise SignalsUnavailable('timeout') from err
            if isinstance(reason, OSError) and reason.errno is not None:
                raise SignalsUnavailable(_strerror(reason)) from err
            raise SignalsUnavailable(str(reason)) from err
        except socket.timeout as err:
            raise SignalsUnavailable('timeout') from err
        except OSError as err:
            raise SignalsUnavailable(_strerror(err)) from err
        try:
            payload = json.loads(body)
        except ValueError as err:
            raise SignalsUnavailable('invalid JSON') from err
        if not isinstance(payload, dict):
            raise SignalsUnavailable('unexpected payload')
        return printable(payload)


def _strerror(err):
    text = (getattr(err, 'strerror', None) or str(err) or 'connection failed').lower()
    return text.split('(', 1)[0].strip()
