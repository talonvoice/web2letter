from flask import Flask, request, jsonify, render_template
import base64
import cffi
import os
import sys
import time

if os.name == 'nt':
    w2l_library = 'libw2lstream.dll'
elif os.name == 'posix':
    if sys.platform == 'darwin':
        w2l_library = 'libw2lstream.dylib'
    else:
        w2l_library = 'libw2lstream.so'
else:
    raise RuntimeError('unsupported OS')

ffi = cffi.FFI()
ffi.cdef('void free(void *);')
with open('w2lstream.h', 'r') as f:
    lines = []
    ifdefs = 0
    for line in f.read().split('\n'):
        line = line.strip()
        if line.startswith('#ifdef'):
            ifdefs += 1
        elif line.startswith('#endif'):
            ifdefs -= 1
        elif ifdefs == 0 and not line.startswith('#'):
            lines.append(line)
    header = '\n'.join(lines)
    ffi.cdef(header)
lib = ffi.dlopen(w2l_library)

good = True
for name in ('feature_extractor.bin', 'acoustic_model.bin', 'tokens.txt'):
    if not os.path.exists(name):
        print('Error: {} not found.'.format(name))
        good = False
if not good:
    sys.exit(1)

encoder_tokens = []
with open('tokens.txt', 'r') as f:
    for line in f:
        encoder_tokens.append(line.strip())

chunk_size = 500 * 16000 // 1000
encoder = lib.w2lstream_new(b'feature_extractor.bin', b'acoustic_model.bin', b'tokens.txt', chunk_size)

def consume_c_text(c_text, sep):
    if not c_text:
        return []
    text = ffi.string(c_text).decode('utf8')
    lib.free(c_text)
    if not text:
        return []
    return text.strip().split(sep)

def w2l_decode(samples, dfa=None):
    start = time.monotonic()
    emit_text = lib.w2lstream_run(encoder, samples, len(samples))
    emit_ms = (time.monotonic() - start) * 1000
    emit = consume_c_text(text, sep=' ')
    if not emit:
        return [], [], emit_ms, 0
    return emit, [], emit_ms, 0

app = Flask('wav2letter')

@app.route('/')
def slash():
    return render_template('index.html')

@app.route('/tokens')
def tokens():
    return jsonify({'tokens': encoder_tokens})

@app.route('/stats')
def stats():
    # TODO: min/max/average emit/decode response times? server loadavg?
    stats = {}
    try:
        l1, l5, l15 = os.getloadavg()
        stats['loadavg'] = [l1, l5, l15]
    except Exception:
        pass
    return jsonify(stats)

# request format (JSON):
# {"cfg": base64(bytes()), "samples": [float samples]}

# response format (JSON):
# {"emit": "some words", "decode": "some words", "emit_ms": 0, "decode_ms": 0}

@app.route('/decode', methods=['POST'])
def recognize():
    j = request.json
    cfg = j.get('cfg', None)
    if cfg:
        if len(cfg) > 0x1000000:
            return jsonify({'error': 'cfg too large'})
        cfg = base64.b64decode(cfg)
    samples = j.get('samples', [])
    if not samples:
        return jsonify({'error': 'not enough samples'})
    if len(samples) > 480000:
        return jsonify({'error': 'too many samples'})
    emit, decode, emit_ms, decode_ms = w2l_decode(samples, cfg)
    return jsonify({'emit': emit, 'decode': decode, 'emit_ms': emit_ms, 'decode_ms': decode_ms})

if __name__ == '__main__':
    app.run(port=5005, debug=True)
