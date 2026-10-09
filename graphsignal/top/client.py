"""HTTP client for the watcher's `/signals` endpoint."""

import json
import socket
import urllib.error
import urllib.request


class SignalsUnavailable(Exception):
    """The endpoint could not be read; `str(exc)` is a short reason for the header."""


class SignalsClient:
    def __init__(self, host, port, timeout=2.0):
        self.host = str(host)
        self.port = int(port)
        self.timeout = float(timeout)

    @property
    def endpoint(self):
        host = f'[{self.host}]' if ':' in self.host else self.host
        return f'{host}:{self.port}'

    @property
    def url(self):
        return f'http://{self.endpoint}/signals'

    def fetch(self):
        """The parsed `/signals` payload, or raise `SignalsUnavailable`."""
        request = urllib.request.Request(self.url, headers={'Accept': 'application/json'})
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
        return payload


def _strerror(err):
    text = (getattr(err, 'strerror', None) or str(err) or 'connection failed').lower()
    return text.split('(', 1)[0].strip()
