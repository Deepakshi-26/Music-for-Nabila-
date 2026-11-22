# 🎂 Nabila, You Make the World Glow
*An original birthday song for Nabila* · C major · 72 BPM (slow ballad) · ~3 min

🎧 **Listen:** [`nabila_birthday_song.mp3`](nabila_birthday_song.mp3)

---

**Intro** *(music box)*

**Verse 1**
Today the sky is shining bright,
the stars came out to say:
a girl so kind, a heart so light
was born upon this day.

**Chorus**
Happy birthday, Nabila,
you make the whole world glow.
Happy birthday, Nabila,
more love than you could know.

**Verse 2**
Your laugh is like a summer song,
your smile can light the room.
May every wish you hold for long
come true and start to bloom.

**Chorus** ×2 *(the last one with harmony)*

**Outro** *(spoken)*
> Happy birthday, Nabila! I hope this year brings you endless smiles,
> beautiful surprises, and every dream you are wishing for.
> You make the world brighter, just by being in it. Happy birthday!

---

### Remaking the song
`make_song.py` builds everything from scratch. A natural female voice (Kokoro neural TTS,
`af_heart`) speaks each word, and the WORLD vocoder holds the vowels and tunes them onto the
melody note by note. The piano, pad, bass, drums and music box are made in numpy.

```bash
pip install numpy pyworld kokoro-onnx soundfile      # plus ffmpeg
mkdir voices && cd voices
curl -LO https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx
curl -LO https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin
cd .. && KOKORO_DIR=voices python3 make_song.py        # VOICE=af_bella etc. to change singer
```
