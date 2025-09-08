
---![Screenshot 2025-06-04 132426](https://github.com/user-attachments/assets/6406d393-ee7d-4445-a5e9-92f07fd43e27)

![Screenshot 2025-06-04 132418](https://github.com/user-attachments/assets/7f423119-4901-406d-a48b-32c3d0bd891f)

---
# 🎭 Chatterbox TTS Pro (SUP3R Edition) Multilingual

**Chatterbox TTS Pro** is a high-quality, customizable text-to-speech (TTS) system enhanced with voice presets, advanced audio effects, and conversation mode, With 23 Languages. This is a fork of [Resemble AI's Chatterbox](https://github.com/resemble-ai/chatterbox), extended with additional audio controls, export options, and a powerful UI via Gradio.

---

## 🚀 Features

- 🎤 **Voice Presets**  
  Save and load voice settings including reference audio for fast reuse.

- 🎛️ **Advanced Audio Effects**  
  Add reverb, echo, pitch shifting, equalizer, 3D spatialization, and noise reduction.

- 🧠 **Conversation Mode**  
  Generate multi-speaker dialogues with different voice presets.

- 📦 **Export Options**  
  Export audio in high/medium/low quality WAV formats.

- 🎚️ **Dynamic Controls**  
  Modify chunk size, temperature, seed, and more to fine-tune output.

---

## 🛠️ Installation

```bash
git clone https://github.com/SUP3RMASS1VE/chatterbox-SUP3R.git
cd chatterbox-SUP3R
python -m venv venv
venv\Scripts\activate
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
uv pip install chatterbox-tts --no-deps --force-reinstall
````

---

## ▶️ Usage

Run the app with:

```bash
python app.py
```

This will launch a Gradio-based interface in your browser where you can:

* Enter text and generate speech
* Upload reference audio
* Enable advanced audio effects
* Save/load/delete voice presets
* Generate conversations between multiple speakers

---

## 📁 File Structure

* `app.py` — Main application logic
* `voice_presets.json` — Stores saved voice preset metadata
* `saved_voices/` — Stores reference audio files
* `exports/` — Output directory for exported audio

---

## 💬 Conversation Mode Format

Use this format to define dialogues:

```
Alice: Hey Bob, how's it going?
Bob: Doing great! Just testing Chatterbox.
Alice: Awesome, it sounds incredible.
```

---

## 🙏 Acknowledgments

Big thanks to the original creators of Chatterbox:
👉 [Resemble AI](https://github.com/resemble-ai) for their groundbreaking work in controllable TTS.

---

## 📜 License

This project is provided under the original license of the upstream [chatterbox](https://github.com/resemble-ai/chatterbox). Check their repository for licensing details.

```

Let me know if you’d like a more developer-focused or user-focused version!
```

