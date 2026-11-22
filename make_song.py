#!/usr/bin/env python3
"""
"Nabila, You Make the World Glow" - an original birthday song.

Generates a full song (instrumental + sung vocals + spoken message) with:
  - Kokoro neural TTS (af_heart female voice) for the words
  - pyworld (WORLD vocoder) to re-pitch and stretch each syllable onto the melody
  - numpy for all instruments (piano, bass, pad, music box, drums) and mixing

Usage:  python3 make_song.py      -> writes nabila_birthday_song.wav / .mp3
Needs:  ffmpeg, numpy, pyworld, kokoro-onnx (+ model files in $KOKORO_DIR)
"""
import os, subprocess, tempfile, wave
import numpy as np
import pyworld as pw

SR = 44100
BPM = 72
BEAT = 60.0 / BPM
BAR = 4 * BEAT
rng = np.random.default_rng(26)

def midi_hz(m): return 440.0 * 2 ** ((m - 69) / 12)
NOTE = {n: i for i, n in enumerate(["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"])}
KEY_SHIFT = int(os.environ.get("KEY_SHIFT", "-5"))   # -5: C major -> G major, a natural range for the voice
def nm(s):  # "C4" -> midi (shifted into the song's key)
    return 12 * (int(s[-1]) + 1) + NOTE[s[:-1]] + KEY_SHIFT

# ----------------------------------------------------------------- song data
# Each melody line is 8 beats: (sung syllable spelling, note, beats)
# Each melody line is 8 beats: (word, [(note, beats) per syllable])
VERSE1 = [
    [("today",[("E4",.5),("G4",1)]),("the",[("G4",.5)]),("sky",[("A4",1)]),("is",[("G4",1)]),("shining",[("E4",1),("D4",.5)]),("bright",[("E4",2.5)])],
    [("the",[("E4",.5)]),("stars",[("F4",1)]),("came",[("F4",.5)]),("out",[("A4",1)]),("to",[("G4",1)]),("say",[("G4",4)])],
    [("uh",[("E4",.5)]),("girl",[("G4",1)]),("so",[("G4",.5)]),("kind",[("C5",1)]),("uh",[("B4",.5)]),("heart",[("A4",1)]),("so",[("G4",.5)]),("light",[("A4",3)])],
    [("was",[("G4",.5)]),("born",[("F4",1)]),("upon",[("E4",.5),("D4",1)]),("this",[("E4",1)]),("day",[("C4",4)])],
]
VERSE2 = [
    [("your",[("E4",.5)]),("laugh",[("G4",1)]),("is",[("G4",.5)]),("like",[("A4",1)]),("uh",[("G4",1)]),("summer",[("E4",1),("D4",.5)]),("song",[("E4",2.5)])],
    [("your",[("E4",.5)]),("smile",[("F4",1)]),("can",[("F4",.5)]),("light",[("A4",1)]),("the",[("G4",1)]),("room",[("G4",4)])],
    [("may",[("E4",.5)]),("every",[("G4",1),("G4",.5)]),("wish",[("C5",1)]),("you",[("B4",.5)]),("hold",[("A4",1)]),("for",[("G4",.5)]),("long",[("A4",3)])],
    [("come",[("G4",.5)]),("true",[("F4",1)]),("and",[("E4",.5)]),("start",[("D4",1)]),("to",[("E4",1)]),("bloom",[("C4",4)])],
]
CHORUS = [
    [("happy",[("C5",.5),("C5",.5)]),("birth",[("A4",1)]),("day",[("C5",1)]),("/nɑːbˈiːlə/",[("D5",.5),("C5",.5),("G4",4)])],
    [("you",[("G4",.5)]),("make",[("A4",1)]),("the",[("G4",.5)]),("whole",[("F4",1)]),("world",[("E4",1)]),("glow",[("D4",4)])],
    [("happy",[("C5",.5),("C5",.5)]),("birth",[("A4",1)]),("day",[("C5",1)]),("/nɑːbˈiːlə/",[("E5",.5),("D5",.5),("C5",4)])],
    [("more",[("A4",.5)]),("love",[("G4",1)]),("than",[("E4",.5)]),("you",[("F4",1)]),("could",[("D4",1)]),("know",[("C4",4)])],
]
# chords per bar (2 bars per line); a bar may hold two chords (2 beats each)
VERSE_CH  = ["C", "Am", "F", "G", "C", "Am", ("F", "G"), "C"]
CHORUS_CH = ["F", "C", "Dm", "G", "F", "Am", ("F", "G"), "C"]
CHORD = {"C": ["C3","E3","G3"], "Am": ["A2","C3","E3"], "F": ["F2","A2","C3"],
         "G": ["G2","B2","D3"], "Dm": ["D3","F3","A3"], "Em": ["E3","G3","B2"]}

SPOKEN = ("Happy birthday, Nabila! I hope this year brings you endless smiles, "
          "beautiful surprises, and every dream you are wishing for. "
          "You make the world brighter, just by being in it. Happy birthday!")

# ----------------------------------------------------------------- helpers
def env_adsr(n, a=0.01, d=0.1, s=0.7, r=0.1):
    t = np.arange(n) / SR
    e = np.ones(n) * s
    na, nd, nr = int(a*SR), int(d*SR), int(r*SR)
    na = min(na, n); e[:na] = np.linspace(0, 1, na)
    nd = min(nd, n - na); e[na:na+nd] = np.linspace(1, s, nd)
    if nr < n: e[-nr:] *= np.linspace(1, 0, nr)
    return e

def add(buf, sig, t):
    i = int(t * SR)
    if i >= len(buf): return
    j = min(len(buf), i + len(sig))
    buf[i:j] += sig[:j - i]

def piano(m, dur, vel=0.5):
    f = midi_hz(m); n = int((dur + 0.6) * SR); t = np.arange(n) / SR
    s = np.zeros(n)
    for h, amp in [(1, 1), (2, .5), (3, .25), (4, .12), (5, .06)]:
        s += amp * np.sin(2*np.pi*f*h*t * (1 + 0.0004*h*h)) * np.exp(-t * (1.2 + 0.9*h))
    s *= np.minimum(1, t / 0.004)
    rel = np.ones(n); k = int(dur * SR)
    if k < n: rel[k:] = np.exp(-np.arange(n-k) / (0.12*SR))
    return vel * s * rel * 0.35

def bell(m, dur, vel=0.4):
    f = midi_hz(m); n = int((dur + 1.5) * SR); t = np.arange(n) / SR
    s = (np.sin(2*np.pi*f*t) + .4*np.sin(2*np.pi*f*2.76*t)*np.exp(-t*3)
         + .2*np.sin(2*np.pi*f*5.4*t)*np.exp(-t*6))
    return vel * s * np.exp(-t * 2.2) * np.minimum(1, t/0.002) * 0.3

def bass(m, dur, vel=0.6):
    f = midi_hz(m); n = int(dur * SR); t = np.arange(n) / SR
    s = np.sin(2*np.pi*f*t) + .3*np.sin(2*np.pi*2*f*t) + .1*np.sin(2*np.pi*3*f*t)
    return vel * s * env_adsr(n, .005, .15, .6, .08) * 0.4

def pad(notes, dur, vel=0.2):
    n = int((dur + 0.5) * SR); t = np.arange(n) / SR; s = np.zeros(n)
    for m in notes:
        f = midi_hz(m + 12)
        for det in (-0.12, 0, 0.12):
            ff = f * 2 ** (det / 12)
            s += np.sin(2*np.pi*ff*t) + .3*np.sin(2*np.pi*2*ff*t) + .1*np.sin(2*np.pi*3*ff*t)
    e = np.minimum(1, t / 0.6) * np.where(t > dur, np.exp(-(t-dur)/0.25), 1)
    return vel * s * e / (3 * len(notes)) * 0.5

def kick():
    n = int(.35*SR); t = np.arange(n)/SR
    f = 50 + 90*np.exp(-t*30)
    return np.sin(2*np.pi*np.cumsum(f)/SR) * np.exp(-t*9) * 0.7

def snare():
    n = int(.25*SR); t = np.arange(n)/SR
    nz = rng.standard_normal(n); nz = np.diff(nz, prepend=0)
    return (0.35*nz*np.exp(-t*18) + 0.3*np.sin(2*np.pi*190*t)*np.exp(-t*25)) * 0.45

def hat():
    n = int(.06*SR); t = np.arange(n)/SR
    nz = np.diff(rng.standard_normal(n), n=2, prepend=[0, 0])
    return nz * np.exp(-t*70) * 0.06

def reverb(x, secs=2.0, wet=0.25):
    n = int(secs*SR); t = np.arange(n)/SR
    ir = rng.standard_normal(n) * np.exp(-t * 3.2)
    ir[:int(0.012*SR)] = 0
    ir /= np.sqrt(np.sum(ir**2))
    L = len(x) + n; N = 1 << int(np.ceil(np.log2(L)))
    y = np.fft.irfft(np.fft.rfft(x, N) * np.fft.rfft(ir, N), N)[:len(x)]
    return x + wet * y * 0.6

def presence(x):
    """Gentle EQ for vocal clarity: cut low rumble, lift 2-5 kHz."""
    N = 1 << int(np.ceil(np.log2(len(x)))); X = np.fft.rfft(x, N); f = np.fft.rfftfreq(N, 1/SR)
    g = np.where(f < 120, (f/120)**2, 1.0) * (1 + 0.6*np.exp(-((f-3200)/1500)**2)) * np.where(f > 9000, 0.7, 1)
    return np.fft.irfft(X * g, N)[:len(x)]

def read_wav(path):
    with wave.open(path) as w:
        d = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float64)
    return d / 32768.0

from kokoro_onnx import Kokoro
KDIR = os.environ.get("KOKORO_DIR", ".")
_kok = None
VOICE = os.environ.get("VOICE", "af_heart")

def resample(x, sr_in, sr_out=SR):
    n_out = int(round(len(x) * sr_out / sr_in))
    X = np.fft.rfft(x); Y = np.zeros(n_out // 2 + 1, complex)
    k = min(len(X), len(Y)); Y[:k] = X[:k]
    return np.fft.irfft(Y, n_out) * (n_out / len(x))

def tts(text, speed=1.0):
    global _kok
    if _kok is None:
        _kok = Kokoro(os.path.join(KDIR, "kokoro-v1.0.onnx"), os.path.join(KDIR, "voices-v1.0.bin"))
    if text.startswith("/"):        # "/phonemes/" -> exact pronunciation (used for the name Nabila)
        s, sr = _kok.create(text.strip("/"), voice=VOICE, speed=speed, lang="en-us", is_phonemes=True)
    else:
        s, sr = _kok.create(text, voice=VOICE, speed=speed, lang="en-us")
    x = resample(np.asarray(s, float), sr)
    nz = np.nonzero(np.abs(x) > 0.01 * np.max(np.abs(x)))[0]
    return x[max(0, nz[0]-300): nz[-1]+300] if len(nz) else x

# ----------------------------------------------------------------- singing
FP = 5.0  # WORLD frame period (ms)
_cache = {}
def analyse(word, nsyl):
    """WORLD-analyse a spoken word and cut it into nsyl syllables at energy dips."""
    key = (word, nsyl)
    if key not in _cache:
        x = tts(word, speed=0.85)
        f0, t = pw.harvest(x, SR, frame_period=FP, f0_floor=90, f0_ceil=600)
        sp = pw.cheaptrick(x, f0, t, SR); ap = pw.d4c(x, f0, t, SR)
        en = np.log(np.sum(sp, axis=1) + 1e-12)
        en = np.convolve(en, np.ones(5)/5, mode="same")
        # harvest marks nearly everything voiced; un-voice breathy/silent frames (p, t, k, s, sh, th, h)
        # so consonants stay crisp instead of being hummed at the note's pitch
        nb = ap.shape[1]; low_ap = ap[:, 1:max(2, nb * 1000 // (SR // 2))].mean(axis=1)
        f0 = f0.copy(); f0[(low_ap > 0.35) | (en < en.max() - 5.0)] = 0
        voiced = np.nonzero((f0 > 0) & (en > en.max() - 3.0))[0]   # loud voiced span (ignores the fading tail)
        v0, v1 = (voiced[0], voiced[-1] + 1) if len(voiced) else (0, len(f0))
        bounds = [0]
        for i in range(1, nsyl):                     # lowest-energy frame near each expected cut
            c = v0 + (v1 - v0) * i // nsyl; w = max(3, (v1 - v0) // (2*nsyl))
            lo, hi = max(bounds[-1] + 2, c - w), min(len(f0) - 2, c + w)
            bounds.append(lo + int(np.argmin(en[lo:hi])) if hi > lo else c)
        bounds.append(len(f0))
        segs = [(f0[a:b], sp[a:b], ap[a:b], en[a:b]) for a, b in zip(bounds[:-1], bounds[1:])]
        _cache[key] = segs
    return _cache[key]

def sing_syl(seg, m, dur, prev_m, vib=True):
    f0, sp, ap, en = seg
    n = len(f0)
    # vowel nucleus = loud voiced frames; only that part is held, consonants keep natural speed
    loud = np.nonzero((f0 > 0) & (en > en.max() - 1.6))[0]
    if len(loud) < 2: loud = np.nonzero(f0 > 0)[0]
    if len(loud) < 2: loud = np.arange(n // 3, max(n // 3 + 2, 2 * n // 3))
    c0, c1 = loud[0], loud[-1] + 1
    onset, coda = np.arange(0, c0), np.arange(c1, n)
    total = int(dur * 1000 / FP * 0.95)
    nv = max(6, total - len(coda))
    core = np.linspace(c0, c1 - 1, nv)
    idx_f = np.concatenate([onset, core, coda]); idx = idx_f.round().astype(int)
    # interpolate spectral frames for a smooth held vowel
    lo = np.floor(idx_f).astype(int); hi = np.minimum(lo + 1, n - 1); fr = (idx_f - lo)[:, None]
    sp2 = sp[lo] * (1 - fr) + sp[hi] * fr; ap2 = ap[lo] * (1 - fr) + ap[hi] * fr
    tt = np.arange(nv) * FP / 1000; cents = np.zeros(nv)
    if prev_m is not None:
        g = min(nv, 10); cents[:g] = (prev_m - m) * 100 * (1 - np.linspace(0, 1, g)) ** 2
    if vib and dur > 0.5:
        cents += 30 * np.clip((tt - 0.3) / 0.35, 0, 1) * np.sin(2*np.pi*5.3*tt)
    f2 = np.where(f0[idx] > 0, midi_hz(m), 0.0)
    f2[len(onset):len(onset)+nv] = midi_hz(m) * 2 ** (cents / 1200)   # hold the vowel fully voiced
    f2[:len(onset)] = np.where(f0[onset] > 0, midi_hz(m), 0)
    y = pw.synthesize(np.ascontiguousarray(f2), np.ascontiguousarray(sp2), np.ascontiguousarray(ap2), SR, FP)
    fade = int(0.025*SR)
    if len(y) > 2*fade: y[-fade:] *= np.linspace(1, 0, fade)
    return y, min(len(onset) * FP / 1000, 0.12)

def sing_lines(buf, lines, start, gain=1.0, transpose=0):
    t = start; prev = None
    for line in lines:
        for word, notes in line:
            segs = analyse(word, len(notes))
            for seg, (note, beats) in zip(segs, notes):
                m = nm(note) + transpose
                y, lead = sing_syl(seg, m, beats * BEAT, prev)
                add(buf, y * gain, max(0, t - lead))
                prev = m; t += beats * BEAT
    return t

# ----------------------------------------------------------------- arrangement
def bar_chords(ch):
    return [(ch, 4)] if isinstance(ch, str) else [(ch[0], 2), (ch[1], 2)]

def comp(inst, chords, start, style):
    t = start
    for ch in chords:
        for name, beats in bar_chords(ch):
            notes = [nm(n) for n in CHORD[name]]
            root = notes[0]
            add(inst["pad"], pad(notes, beats*BEAT, 0.22 if style != "intro" else 0.18), t)
            if style in ("intro", "verse"):           # flowing piano arpeggio
                pat = [notes[0]+12, notes[1]+12, notes[2]+12, notes[1]+24, notes[2]+12, notes[1]+12, notes[0]+24, notes[2]+12]
                for k in range(int(beats*2)):
                    add(inst["piano"], piano(pat[k % 8], BEAT*0.6, 0.38 if k % 2 == 0 else 0.28), t + k*BEAT/2)
            else:                                      # chorus: chords + groove
                for k in range(int(beats)):
                    v = 0.42 if k % 2 == 0 else 0.32
                    for nn in notes: add(inst["piano"], piano(nn+12, BEAT*0.9, v*0.7), t + k*BEAT)
                    add(inst["piano"], piano(notes[1]+24, BEAT*0.4, 0.2), t + k*BEAT + BEAT/2)
            if style != "intro":
                for k in range(0, int(beats), 2):
                    add(inst["bass"], bass(root - 12 if root >= nm("C3") else root, BEAT*1.8), t + k*BEAT)
                    add(inst["bass"], bass(root + 7 - 12 if root >= nm("C3") else root + 7, BEAT*0.45, 0.4), t + (k+1.5)*BEAT)
            if style == "chorus":
                for k in range(int(beats)):
                    bt = t + k*BEAT
                    if k % 2 == 0: add(inst["drums"], kick(), bt)
                    else: add(inst["drums"], snare(), bt)
                    add(inst["drums"], hat(), bt); add(inst["drums"], hat()*0.7, bt + BEAT/2)
                add(inst["drums"], kick()*0.6, t + 2.5*BEAT) if beats == 4 else None
            elif style == "verse":
                for k in range(int(beats)):
                    add(inst["drums"], hat()*0.6, t + k*BEAT + BEAT/2)
                    if k == 0: add(inst["drums"], kick()*0.7, t)
            t += beats * BEAT
    return t

def music_box(buf, lines, start, transpose=12):
    t = start
    for line in lines:
        for _, notes in line:
          for note, beats in notes:
            add(buf, bell(nm(note) + transpose, beats*BEAT), t); t += beats*BEAT
    return t

def main():
    print("Synthesising spoken message ...")
    speech = tts(SPOKEN, speed=0.95)
    speech_len = len(speech) / SR
    outro_bars = int(np.ceil((speech_len + 2.5) / BAR))

    sections = (["intro"]*4 + ["verse"]*8 + ["chorus"]*8 + ["inter"]*2 +
                ["verse"]*8 + ["chorus"]*8 + ["chorus"]*8 + ["outro"]*outro_bars)
    length = len(sections) * BAR + 4
    inst = {k: np.zeros(int(length * SR)) for k in ["piano", "pad", "bass", "drums", "bell", "vox", "harm", "speech"]}

    t = 0.0
    print("Arranging instruments ...")
    # Intro: music box plays the chorus tune over soft arpeggios
    music_box(inst["bell"], CHORUS[:2], t)
    t = comp(inst, ["C", "Am", "F", "G"], t, "intro")
    v1 = t; t = comp(inst, VERSE_CH, t, "verse")
    c1 = t; t = comp(inst, CHORUS_CH, t, "chorus")
    music_box(inst["bell"], [CHORUS[1]], t)
    t = comp(inst, ["Am", "G"], t, "intro")
    v2 = t; t = comp(inst, VERSE_CH, t, "verse")
    c2 = t; t = comp(inst, CHORUS_CH, t, "chorus")
    c3 = t; t = comp(inst, CHORUS_CH, t, "chorus")
    outro = t
    music_box(inst["bell"], CHORUS[2:], outro)
    comp(inst, (["F", "G", "C", "C"] * 4)[:outro_bars], outro, "intro")
    add(inst["bell"], bell(nm("C6"), 3, 0.5), outro + outro_bars*BAR - BAR)

    print("Singing (this takes a minute) ...")
    sing_lines(inst["vox"], VERSE1, v1)
    sing_lines(inst["vox"], CHORUS, c1)
    sing_lines(inst["vox"], VERSE2, v2)
    sing_lines(inst["vox"], CHORUS, c2)
    sing_lines(inst["vox"], CHORUS, c3)
    # harmony on the last chorus (a third-ish below, diatonic)
    HARM = {"C5":"A4","D5":"B4","E5":"C5","A4":"F4","G4":"E4","F4":"D4","E4":"C4","D4":"B3","C4":"E3","B4":"G4"}
    harm = [[(w, [(HARM.get(n, n), b) for n, b in ns]) for w, ns in line] for line in CHORUS]
    sing_lines(inst["harm"], harm, c3, gain=0.4)
    add(inst["speech"], speech, outro + 1.0)

    print("Mixing ...")
    vox = reverb(presence(inst["vox"] + inst["harm"]), 1.6, 0.16)
    vox = 1.6 * vox / (np.max(np.abs(vox)) + 1e-9) * 0.6
    spk = 0.6 * reverb(inst["speech"], 1.5, 0.15) / (np.max(np.abs(inst["speech"])) + 1e-9)
    music = (0.7*reverb(inst["piano"], 1.8, 0.25) + 0.7*inst["pad"] + 0.9*inst["bass"]
             + 0.35*inst["drums"] + 0.8*reverb(inst["bell"], 2.5, 0.4))
    # sidechain: the music dips whenever the voice is present, so every word sits on top
    k = int(0.06 * SR)
    lvl = np.sqrt(np.convolve((inst["vox"] + inst["speech"])**2, np.ones(k)/k, mode="same"))
    lvl = np.convolve(lvl, np.ones(k)/k, mode="same")
    lvl /= np.percentile(lvl[lvl > 1e-4], 90) + 1e-9
    duck = 1 - 0.45 * np.clip(lvl, 0, 1)
    mix = music * duck + vox + spk
    end = int((outro + outro_bars*BAR + 2.5) * SR)
    mix = mix[:end]
    global SECTION_TIMES, VOX_STEM
    SECTION_TIMES = dict(v1=v1, c1=c1, v2=v2, c2=c2, c3=c3, outro=outro)
    VOX_STEM = inst["vox"][:end]
    mix[-int(2.5*SR):] *= np.linspace(1, 0, int(2.5*SR))
    mix /= np.max(np.abs(mix)) + 1e-9
    mix = np.tanh(mix * 1.3) / np.tanh(1.3) * 0.95

    out = "nabila_birthday_song"
    pcm = (mix * 32767).astype(np.int16)
    with wave.open(out + ".wav", "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR); w.writeframes(pcm.tobytes())
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", out + ".wav", "-b:a", "192k", out + ".mp3"], check=True)
    print(f"Done: {out}.mp3  ({len(mix)/SR:.0f} s)")

if __name__ == "__main__":
    main()
