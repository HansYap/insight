# Insight

A privacy-first, fully local ambient intelligence system, its a personal AI assistant that lives at home and grows smarter over time through human-in-the-loop feedback.

---

## What Is This?

Insight is a distributed vision pipeline that watches a fixed scene, recognises what is happening, and builds a personal memory of your activities over time. It starts by asking you what it sees. 
Over time, it learns to recognise familiar situations without asking. Furthermore, the longer it runs, the more it knows about your specific life by learning from personal data.

The core loop:

```
See something → uncertain → ask you → you label it →
store the embedding → next time, recognise without asking
```

This project uses **retrieval-augmented recognition with human-in-the-loop feedback**, a form of interactive continual learning that sidesteps catastrophic forgetting by keeping model weights frozen and treating learning as additive memory rather than weight updates.

---

## Architecture

<img width="735" height="665" alt="image" src="https://github.com/user-attachments/assets/7a4e5bb7-100c-464c-9324-78d8a6034076" />


---

## The Vector Space

Every recognised event is stored as a 390-dimensional vector:

```
V_final = [1.0 × V_vision (384-dim)] + [8.0 × V_motion (6-dim)]
```

**V_vision (384-dim):** Sentence transformer embedding (`all-MiniLM-L6-v2`) of the Florence-2 (Microsoft's open source vision model) scene description, it captures semantic content (what is in the frame and what is happening).

**V_motion (6-dim):** Farneback optical flow descriptor over a 10-frame rolling buffer, computed entirely on the Pi. It acts as a proxy for encoding motion in otherwise static images (inspired by DeepMind Atari).

| Dimension | Feature | Description |
|---|---|---|
| 0 | `mean_magnitude` | Average pixel displacement: overall motion intensity |
| 1 | `std_magnitude` | Motion variance: localised vs distributed movement |
| 2 | `directionality` | Circular variance of flow vectors: how ordered the motion is |
| 3 | `coverage_ratio` | Fraction of frame with significant motion |
| 4 | `dominant_sin` | Y-component of mean flow direction |
| 5 | `dominant_cos` | X-component of mean flow direction |

The 8.0× weight on motion reflects the intuition that two scenes with identical visual content but different motion profiles (sitting vs. exercising at a desk) are meaningfully different events and should not collapse to the same memory address.

Normalization constants are derived from empirically collected baseline data and stored in `config/vmotion_norm.yaml`. **Camera repositioning invalidates these constants**, thus recalibration and rebuilding ChromaDB is required if the camera moves.

---

## The Memory System

At inference time, the pipeline queries ChromaDB with the current frame's 390-dim vector and retrieves the nearest stored memory by cosine similarity.

```
similarity ≥ 0.9   →   Confident match. Log automatically.
sim < 0.92   →   No match. Ask the user to label from scratch.
```

When the user provides a label, it is stored immediately and propagated to similar embeddings (cosine similarity ≥ 0.9) in the existing database. This means a single label can silently update many related memories.

Training data accumulates from three sources:

| Source | Description |
|---|---|
| `auto` | Confident ChromaDB matches (sim ≥ 0.9) written at inference time |
| `user` | Human labels provided via the inbox dialogue |
| `propagated` | Similarity sweep against newly labelled embeddings |

---

## Hardware

| Component | Role |
|---|---|
| Raspberry Pi 5 (4GB RAM) | Always-on edge inference: YOLOv8, optical flow, motion trigger |
| Logitech C920 (1080p USB) | Primary camera |
| ThinkPad P53s (1TB) | Processing backend: Florence-2, ChromaDB, SQLite, FastAPI |

The Pi and ThinkPad communicate over LAN. The Pi sends JPEG frames to the ThinkPad's Florence-2 FastAPI server and receives natural language descriptions + embeddings in return.

---

## Privacy

Insight is designed privacy-first by constraint since it operates in my personal space.

- All inference runs locally: no API calls, no cloud services
- No audio collection
- Video frames are processed in memory and discarded; only embeddings and descriptions persist
- The SQLite database and ChromaDB store are local and not synced anywhere
- The system has no network egress beyond the LAN between the Pi and ThinkPad

The models are frozen open-source weights. The memory that makes this personal is generated entirely from own labelling.

---

## Why This Architecture

Standard continual learning approaches retrain model weights incrementally, which causes **catastrophic forgetting** — the model forgets old tasks as it learns new ones. Fine-tuning is also computationally infeasible due to lack of proper hardware.
Insight avoids this entirely by keeping all model weights frozen and treating learning as **additive memory** in a vector database. The `all-MiniLM-L6-v2` embedding space is fixed permanently. New knowledge is a new point in that space, not a change to the space itself. The cost is that the system cannot learn visual features that fall outside Florence-2's training distribution. The benefit is that memory is perfectly stable, always inspectable, and can be pruned, corrected, or rebuilt from scratch at any time.
Also, I chose this architecture because I had additional hardware lying around and wanted to avoid wasting money.

---

## References

- Mnih, V. et al. (2013). *Playing Atari with Deep Reinforcement Learning.*
- Farnebäck, G. (2003). *Two-Frame Motion Estimation Based on Polynomial Expansion.* 
