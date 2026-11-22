#!/usr/bin/env python3
"""
"Nabila, You Make the World Glow" - an original birthday song.

Generates a full song (instrumental + sung vocals + spoken message) with:
  - espeak-ng + MBROLA (us1 female voice) for the words
  - pyworld (WORLD vocoder) to re-pitch and stretch each syllable onto the melody
  - numpy for all instruments (piano, bass, pad, music box, drums) and mixing

Usage:  python3 make_song.py      -> writes nabila_birthday_song.wav / .mp3
Needs:  espeak-ng, mbrola-us1, ffmpeg, numpy, pyworld
"""
import os, subprocess, tempfile, wave
import numpy as np
import pyworld as pw

SR = 44100
BPM = 96
BEAT = 60.0 / BPM
BAR = 4 * BEAT
rng = np.random.default_rng(26)

def midi_hz(m): return 440.0 * 2 ** ((m - 69) / 12)
NOTE = {n: i for i, n in enumerate(["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"])}
def nm(s):  # "C4" -> midi
    return 12 * (int(s[-1]) + 1) + NOTE[s[:-1]]

# ----------------------------------------------------------------- song data
# Each melody line is 8 beats: (sung syllable spelling, note, beats)
VERSE1 = [
    [("too","E4",.5),("day","G4",1),("the","G4",.5),("sky","A4",1),("is","G4",1),("shy","E4",1),("ning","D4",.5),("bright","E4",2.5)],
    [("the","E4",.5),("stars","F4",1),("came","F4",.5),("out","A4",1),("to","G4",1),("say","G4",4)],
    [("a","E4",.5),("girl","G4",1),("so","G4",.5),("kind","C5",1),("a","B4",.5),("heart","A4",1),("so","G4",.5),("light","A4",3)],
    [("was","G4",.5),("born","F4",1),("up","E4",.5),("on","D4",1),("this","E4",1),("day","C4",4)],
]
VERSE2 = [
    [("your","E4",.5),("laugh","G4",1),("is","G4",.5),("like","A4",1),("a","G4",1),("sum","E4",1),("mer","D4",.5),("song","E4",2.5)],
    [("your","E4",.5),("smile","F4",1),("can","F4",.5),("light","A4",1),("the","G4",1),("room","G4",4)],
    [("may","E4",.5),("ev","G4",1),("ree","G4",.5),("wish","C5",1),("you","B4",.5),("hold","A4",1),("for","G4",.5),("long","A4",3)],
    [("come","G4",.5),("true","F4",1),("and","E4",.5),("start","D4",1),("to","E4",1),("bloom","C4",4)],
]
CHORUS = [
    [("hap","C5",.5),("pee","C5",.5),("birth","A4",1),("day","C5",1),("nah","D5",.5),("bee","C5",.5),("lah","G4",4)],
    [("you","G4",.5),("make","A4",1),("the","G4",.5),("whole","F4",1),("world","E4",1),("glow","D4",4)],
    [("hap","C5",.5),("pee","C5",.5),("birth","A4",1),("day","C5",1),("nah","E5",.5),("bee","D5",.5),("lah","C5",4)],
    [("more","A4",.5),("love","G4",1),("than","E4",.5),("you","F4",1),("could","D4",1),("know","C4",4)],
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

def read_wav(path):
    with wave.open(path) as w:
        d = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float64)
    return d / 32768.0

TMP = tempfile.mkdtemp()
def tts(text, speed=150, pitch=50):
    raw = os.path.join(TMP, "raw.wav"); out = os.path.join(TMP, "out.wav")
    subprocess.run(["espeak-ng", "-v", "mb-us1", "-s", str(speed), "-p", str(pitch), "-w", raw, text],
                   check=True, capture_output=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", raw, "-ar", str(SR), "-ac", "1", out], check=True)
    x = read_wav(out)
    nz = np.nonzero(np.abs(x) > 0.01)[0]
    return x[max(0, nz[0]-200): nz[-1]+200] if len(nz) else x

# ----------------------------------------------------------------- singing
FP = 5.0  # WORLD frame period (ms)
_cache = {}
def analyse(word):
    if word not in _cache:
        x = tts(word, speed=130)
        f0, t = pw.harvest(x, SR, frame_period=FP, f0_floor=70, f0_ceil=500)
        sp = pw.cheaptrick(x, f0, t, SR)
        ap = pw.d4c(x, f0, t, SR)
        _cache[word] = (f0, sp, ap)
    return _cache[word]

def sing(word, m, dur, prev_m=None, vib=True):
    """Return (audio, lead_in_seconds) of `word` sung at midi note m for dur seconds."""
    f0, sp, ap = analyse(word)
    voiced = np.nonzero(f0 > 0)[0]
    if len(voiced) == 0:
        voiced = np.arange(len(f0)//3, 2*len(f0)//3)
    v0, v1 = voiced[0], voiced[-1] + 1
    onset = np.arange(0, v0); coda = np.arange(v1, len(f0))
    onset_s = len(onset) * FP / 1000
    total = int(dur * 1000 / FP * 0.94)             # slight gap before the next note
    nv = max(4, total - len(coda))
    mid = np.linspace(v0, v1 - 1, nv).round().astype(int)
    idx = np.concatenate([onset, mid, coda])
    sp2, ap2 = sp[idx].copy(), ap[idx].copy()
    target = np.full(len(idx), midi_hz(m))
    tt = np.arange(nv) * FP / 1000
    cents = np.zeros(nv)
    if prev_m is not None:                             # small glide from the previous note
        g = min(nv, 9)
        cents[:g] = (prev_m - m) * 100 * (1 - np.linspace(0, 1, g)) ** 2
    if vib and dur > 0.45:                             # gentle delayed vibrato
        ramp = np.clip((tt - 0.25) / 0.3, 0, 1)
        cents += 28 * ramp * np.sin(2*np.pi*5.4*tt)
    f2 = np.zeros(len(idx))
    f2[len(onset):len(onset)+nv] = target[:nv] * 2 ** (cents / 1200)
    orig_v = f0[idx] > 0
    f2[len(onset):len(onset)+nv][~orig_v[len(onset):len(onset)+nv]] = 0
    y = pw.synthesize(np.ascontiguousarray(f2), np.ascontiguousarray(sp2), np.ascontiguousarray(ap2), SR, FP)
    fade = int(0.02*SR)
    if len(y) > 2*fade: y[-fade:] *= np.linspace(1, 0, fade)
    return y, min(onset_s, 0.09)

def sing_lines(buf, lines, start, gain=1.0, transpose=0):
    t = start; prev = None
    for line in lines:
        for word, note, beats in line:
            m = nm(note) + transpose
            y, lead = sing(word, m, beats * BEAT, prev)
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
        for _, note, beats in line:
            add(buf, bell(nm(note) + transpose, beats*BEAT), t); t += beats*BEAT
    return t

def main():
    print("Synthesising spoken message ...")
    speech = tts(SPOKEN, speed=138, pitch=55)
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
    harm = [[(w, HARM.get(n, n), b) for w, n, b in line] for line in CHORUS]
    sing_lines(inst["harm"], harm, c3, gain=0.55)
    add(inst["speech"], speech, outro + 1.0)

    print("Mixing ...")
    vox = reverb(inst["vox"] + inst["harm"], 2.2, 0.32)
    mix = (0.9*reverb(inst["piano"], 1.8, 0.25) + 0.7*inst["pad"] + 0.9*inst["bass"]
           + 0.55*inst["drums"] + 0.8*reverb(inst["bell"], 2.5, 0.4)
           + 1.25*vox/ (np.max(np.abs(vox)) + 1e-9) * 0.55
           + 0.6*reverb(inst["speech"], 1.5, 0.15) / (np.max(np.abs(inst["speech"])) + 1e-9))
    # duck the music under the spoken message
    duck = np.ones_like(mix); s0 = int((outro + 0.5)*SR); s1 = int((outro + 1.5 + speech_len)*SR)
    music = mix - 0.6*reverb(inst["speech"], 1.5, 0.15) / (np.max(np.abs(inst["speech"])) + 1e-9)
    duck[s0:s1] = 0.55
    mix = music*duck + (mix - music)
    end = int((outro + outro_bars*BAR + 2.5) * SR)
    mix = mix[:end]
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
