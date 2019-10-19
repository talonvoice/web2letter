from flask import Flask, request, jsonify, render_template
import base64
import cffi
import os
import sys
import time

if os.name == 'nt':
    w2l_library = 'libw2l.dll'
elif os.name == 'posix':
    if sys.platform == 'darwin':
        w2l_library = 'libw2l.dylib'
    else:
        w2l_library = 'libw2l.so'
else:
    raise RuntimeError('unsupported OS')

ffi = cffi.FFI()
ffi.cdef('void free(void *);')
with open('w2l.h', 'r') as f:
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
for name in ('acoustic.bin', 'tokens.txt', 'lm-ngram.bin', 'lexicon.txt'):
    if not os.path.exists(name):
        print('Error: {} not found.'.format(name))
        good = False
if not good:
    sys.exit(1)

if not os.path.exists('lexicon_flat.bin'):
    lib.w2l_make_flattrie(b'tokens.txt', b'lm-ngram.bin',
                          b'lexicon.txt', b'lexicon_flat.bin')

# set up decode options
decode_opts = ffi.new('w2l_decode_options *')
decode_opts.beamsize = 183
decode_opts.beamthresh = 23.530
decode_opts.lmweight = 1.30
decode_opts.wordscore = 0.5
decode_opts.unkweight = -float('Inf')
decode_opts.logadd = False
decode_opts.silweight = 0

dfa_opts = ffi.new('w2l_dfa_decode_options *')
dfa_opts.command_score = 0.5
dfa_opts.rejection_threshold = 0.55
dfa_opts.rejection_window_frames = 8
dfa_opts.debug = False
# end decode opts

encoder_tokens = []
with open('tokens.txt', 'r') as f:
    for line in f:
        encoder_tokens.append(line.strip())

encoder = lib.w2l_engine_new(b'acoustic.bin', b'tokens.txt')
decoder = lib.w2l_decoder_new(encoder, b'lm-ngram.bin', b'lexicon.txt', b'lexicon_flat.bin', decode_opts)

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
    emission = lib.w2l_engine_process(encoder, samples, len(samples))
    emit_ms = (time.monotonic() - start) * 1000

    emit_text = lib.w2l_emission_text(emission)
    emit = consume_c_text(emit_text, sep=' ')
    if not emit:
        return [], [], emit_ms, 0

    start = time.monotonic()
    if dfa:
        dfa_node = ffi.cast('w2l_dfa_node *', ffi.from_buffer(dfa))
        decode_text = lib.w2l_decoder_dfa(encoder, decoder, emission, dfa_node, dfa_opts)
    else:
        decode_result = lib.w2l_decoder_decode(decoder, emission)
        decode_text = lib.w2l_decoder_result_words(decoder, decode_result)
        lib.w2l_decoderesult_free(decode_result)
    decode_ms = (time.monotonic() - start) * 1000
    lib.w2l_emission_free(emission)

    decode = consume_c_text(decode_text, sep=' ')
    return emit, decode, emit_ms, decode_ms

app = Flask('wav2letter')

@app.route('/')
def slash():
    # TODO: ship webrtcvad here and let people play with it interactively?
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
