from flask import Flask, request, jsonify, render_template
import base64
import bson
import cffi
import io
import os
import soundfile as sf
import sys
import time
import zlib

if os.name == 'nt':
    libext = 'dll'
elif os.name == 'posix':
    if sys.platform == 'darwin':
        libext = 'dylib'
    else:
        libext = 'so'
else:
    raise RuntimeError('unsupported OS')
libw2l = 'lib/libw2l.{}'.format(libext)
libw2ldecode = 'lib/libw2ldecode.{}'.format(libext)

def read_header(path):
    with open(path, 'r') as f:
        lines = []
        ifdefs = 0
        for line in f.read().split('\n'):
            line = line.strip()
            line = line.replace('__attribute__((packed))', '')
            if line.startswith('#ifdef'):
                ifdefs += 1
            elif line.startswith('#endif'):
                ifdefs -= 1
            elif ifdefs == 0 and not line.startswith('#'):
                lines.append(line)
        return '\n'.join(lines)

common_h = read_header('include/w2l_common.h')
encode_h = read_header('include/w2l_encode.h')
decode_h = read_header('include/w2l_decode.h')

ffi = cffi.FFI()
ffi.cdef('void free(void *);')
ffi.cdef(common_h)
ffi.cdef(encode_h)
lib = ffi.dlopen(libw2l)

decodeffi = cffi.FFI()
decodeffi.cdef(common_h)
decodeffi.cdef(decode_h)
decodelib = decodeffi.dlopen(libw2ldecode)

good = True
for name in ('model/acoustic.b2l', 'model/tokens.txt', 'model/lm-ngram.bin', 'model/lexicon.txt'):
    if not os.path.exists(name):
        print('Error: {} not found.'.format(name))
        good = False
if not good:
    sys.exit(1)

# set up decode options
opts = decodeffi.new('w2l_decode_options *')

opts.beamsize = 183
opts.beamsizetoken = 100
opts.beamthresh = 23.530
opts.lmweight = 1.30
opts.wordscore = 0.5
opts.unkweight = -float('Inf')
opts.logadd = False
opts.silweight = 0

c_criterion = ffi.new('char[]', b'ctc') # FIXME
opts.criterion = c_criterion

opts.command_score = 0.5
opts.rejection_threshold = 0.55
opts.rejection_window_frames = 8
opts.debug = False
# end decode opts

encoder_tokens = []
with open('model/tokens.txt', 'r') as f:
    for line in f:
        encoder_tokens.append(line.strip())
tokens = '\n'.join(encoder_tokens)

encoder = lib.w2l_engine_new()
if not encoder or not lib.w2l_engine_load_b2l(encoder, b'model/acoustic.b2l'):
    raise RuntimeError('failed to load model')
decoder = decodelib.w2l_decoder_new(tokens.encode('utf8'), b'model/lm-ngram.bin', b'model/lexicon.txt', opts)

c_trie_path = b'model/lexicon-flat.bin'
if not decodelib.w2l_decoder_load_trie(decoder, c_trie_path):
    if not decodelib.w2l_decoder_make_trie(decoder, c_trie_path):
        raise RuntimeError('trie creation failed')
    if not decodelib.w2l_decoder_load_trie(decoder, c_trie_path):
        raise RuntimeError('trie load failed in w2l_decoder_load_trie()')

def consume_c_text(c_text, sep):
    if not c_text:
        return []
    text = ffi.string(c_text).decode('utf8')
    lib.free(c_text)
    if not text:
        return []
    return text.strip().strip(sep).split(sep)

def w2l_decode(samples, dfa=None):
    c_samples = ffi.new('float[]', samples)
    start = time.monotonic()
    emission = lib.w2l_engine_forward(encoder, c_samples, len(samples))
    emission = decodeffi.cast('w2l_emission *', emission)
    emit_ms = (time.monotonic() - start) * 1000

    emit_text = decodelib.w2l_decoder_greedy(decoder, emission)
    emit = consume_c_text(emit_text, sep='|')
    if not emit:
        lib.free(emission)
        return '', [], emit_ms, 0

    start = time.monotonic()
    if dfa:
        dfa_node = ffi.cast('w2l_dfa_node *', ffi.from_buffer(dfa))
        decode_text = decodelib.w2l_decoder_dfa(decoder, emission, dfa_node, len(dfa))
    else:
        decode_text = decodelib.w2l_decoder_decode(decoder, emission)
    decode_ms = (time.monotonic() - start) * 1000
    lib.free(emission)

    decode = consume_c_text(decode_text, sep=' ')
    return ' '.join(emit), decode, emit_ms, decode_ms

app = Flask('web2letter')

@app.route('/')
def slash():
    return render_template('index.html')

# TODO: use /info in web UI?
@app.route('/info')
def info():
    return jsonify({'version': 3, 'tokens': encoder_tokens})

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
    if request.content_type == 'application/bson':
        data = request.data
        if request.content_encoding == 'gzip':
            data = zlib.decompress(data)
        j = bson.loads(data)
        cfg = j.get('cfg', None)
    else:
        j = request.json
        cfg = j.get('cfg', None)
        if cfg:
            cfg = base64.b64decode(cfg)
    if cfg and len(cfg) > 0x1000000:
        return jsonify({'error': 'cfg too large'})

    if j.get('version', 0) > 1 and 'flac' in j:
        flacsamples, samplerate = sf.read(io.BytesIO(j['flac']))
        samples = flacsamples.tolist()
    else:
        samples = j.get('samples', [])
    if not samples:
        return jsonify({'error': 'not enough samples'})
    if len(samples) > 480000:
        return jsonify({'error': 'too many samples'})
    emit, decode, emit_ms, decode_ms = w2l_decode(samples, cfg)
    return jsonify({'emit': emit, 'decode': decode, 'emit_ms': emit_ms, 'decode_ms': decode_ms})

if __name__ == '__main__':
    app.run(port=5005, debug=True)
