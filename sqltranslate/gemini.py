"""Gemini native mapping with request caching and bounded transport retries."""
import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from .evaluation import read_json, write_json, validate_mapping
from .fewshot import build_messages

TRANSIENT_HTTP = {429, 500, 502, 503, 504}


class TemporaryAPIError(RuntimeError):
    """Pause a batch without treating service availability as model accuracy."""


class APIConfigurationError(RuntimeError):
    """Authentication, endpoint or model configuration needs attention."""


def native_body(source, target, examples, max_tokens=8192):
    messages = build_messages(source, target, examples)
    return {
        'systemInstruction': {'parts': [{'text': messages[0]['content']}]},
        'contents': [{'role': 'model' if m['role'] == 'assistant' else 'user',
                      'parts': [{'text': m['content']}]} for m in messages[1:]],
        'generationConfig': {'temperature': 0, 'maxOutputTokens': max_tokens,
                             'responseMimeType': 'application/json'}
    }


def parse_mapping(payload, source, target):
    choices = payload.get('candidates', [])
    if not choices:
        raise ValueError('No candidates; promptFeedback=' + str(payload.get('promptFeedback', {})))
    candidate = choices[0]
    if candidate.get('finishReason') != 'STOP':
        raise ValueError('Incomplete generation: ' + str(candidate.get('finishReason')))
    text = ''.join(part.get('text', '') for part in candidate.get('content', {}).get('parts', [])
                   if not part.get('thought', False))
    return validate_mapping(json.loads(text), source, target)


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    write_json(temporary, value)
    temporary.replace(path)


class GeminiMatcher:
    def __init__(self, root, model, examples, logdir, key_env='GEMINI_API_KEY', max_tokens=8192,
                 max_attempts=1, min_interval=0, sleep=None, clock=None, notify=None):
        if max_attempts < 1 or min_interval < 0:
            raise ValueError('Invalid retry or interval settings')
        self.model, self.examples, self.key_env = model, examples, key_env
        self.logdir = Path(logdir)
        self.logdir.mkdir(parents=True, exist_ok=True)
        self.max_tokens, self.max_attempts, self.min_interval = max_tokens, max_attempts, min_interval
        self.sleep, self.clock = sleep or time.sleep, clock or time.monotonic
        self.notify = notify or (lambda text: None)
        self.last_start = None

    def __call__(self, pair_id, source, target):
        endpoint = 'https://generativelanguage.googleapis.com/v1beta/models/' + self.model + ':generateContent'
        body = native_body(source, target, self.examples, self.max_tokens)
        encoded = json.dumps(body, ensure_ascii=False).encode()
        digest = hashlib.sha256(endpoint.encode() + encoded).hexdigest()
        path = self.logdir / (digest + '.json')
        if path.exists():
            cached = read_json(path)
            if cached.get('status') == 'success':
                self.notify('使用已缓存的 mapping')
                return validate_mapping(cached['mapping'], source, target)
            if not cached.get('retryable', cached.get('http_status') in TRANSIENT_HTTP):
                error_type = APIConfigurationError if cached.get('http_status') else ValueError
                raise error_type('Cached non-retryable failure; see ' + str(path))
        key = os.environ.get(self.key_env)
        if not key:
            raise ValueError('Set ' + self.key_env)
        attempts = sorted(self.logdir.glob(digest + '.attempt_*.json'))
        sequence = max([int(p.stem.rsplit('_', 1)[1]) for p in attempts] or [0])
        for attempt in range(self.max_attempts):
            if self.last_start is not None:
                delay = self.min_interval - (self.clock() - self.last_start)
                if delay > 0:
                    self.sleep(delay)
            self.last_start = self.clock()
            log = {'pair_id': pair_id, 'model': self.model, 'endpoint': endpoint,
                   'request': body, 'request_sha256': digest,
                   'attempt': sequence + attempt + 1,
                   'example_pair_ids': [e['pair_id'] for e in self.examples]}
            start = self.clock()
            retry_after, daily_quota = 0, False
            failure = None
            try:
                request = urllib.request.Request(endpoint, data=encoded,
                    headers={'x-goog-api-key': key, 'Content-Type': 'application/json'})
                with urllib.request.urlopen(request, timeout=120) as response:
                    payload = json.load(response)
                log['response'] = payload
                log['usage'] = payload.get('usageMetadata', {})
                log['response_model'] = payload.get('modelVersion')
                mapping = parse_mapping(payload, source, target)
                log.update(status='success', mapping=mapping, retryable=False)
            except urllib.error.HTTPError as exc:
                raw = exc.read().decode('utf-8', errors='replace').replace(key, '[REDACTED]')
                log.update(status='failure', http_status=exc.code,
                           response_content_type=exc.headers.get('Content-Type'),
                           response_body=raw[:4000], retryable=exc.code in TRANSIENT_HTTP)
                try:
                    retry_after = max(0, float(exc.headers.get('Retry-After', '0')))
                except ValueError:
                    pass
                daily_quota = exc.code == 429 and any(s in raw.lower() for s in ('perday', 'per_day', 'daily', 'limit: 0', '"quotavalue": "0"'))
                failure = RuntimeError(f'Native API HTTP {exc.code}: {raw[:600]}')
            except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
                log.update(status='failure', retryable=True, error=str(exc).replace(key, '[REDACTED]'))
                failure = exc
            except Exception as exc:
                log.update(status='failure', retryable=False, error=str(exc).replace(key, '[REDACTED]'))
                failure = exc
            finally:
                log['elapsed_seconds'] = self.clock() - start
                atomic_json(self.logdir / f'{digest}.attempt_{sequence + attempt + 1:04d}.json', log)
                atomic_json(path, log)
            if log['status'] == 'success':
                return mapping
            if not log['retryable']:
                error_type = APIConfigurationError if log.get('http_status') else ValueError
                raise error_type(log.get('error', str(failure))) from None
            if daily_quota or retry_after > 120 or attempt + 1 == self.max_attempts:
                raise TemporaryAPIError(str(failure)) from None
            delay = max(retry_after, 20 * 2 ** attempt)
            self.notify(f'临时 API 错误；{delay:.0f} 秒后重试（最多 {self.max_attempts} 次请求）')
            self.sleep(delay)
