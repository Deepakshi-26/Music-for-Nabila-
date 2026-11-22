# 🎂 Nabila, You Make the World Glow
*An original birthday song for Nabila* · C major · 96 BPM · ~2 min 12 s

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
`make_song.py` builds everything from scratch: a female voice (espeak-ng + MBROLA `us1`)
is re-pitched note by note onto the melody with the WORLD vocoder, over piano, pad,
bass, drums and music box made in numpy. Edit the lyrics or melody in the script and run:

```bash
sudo apt-get install espeak-ng mbrola-us1 ffmpeg
pip install numpy pyworld
python3 make_song.py
```
