# Kangaroo SysID Scripts

Diese Skripte implementieren eine gradientenbasierte System-Identifikation (SysID)
für den Kangaroo-Arm mit MuJoCo MJX als differenzierbarem Simulator.

## Methode: Gradient-Based System Identification

### Problemformulierung

Sei $\theta \in \mathbb{R}^d$ ein Vektor physikalischer Parameter
(hier: `armature`, `damping`, `frictionloss` des Ellenbogengelenks).

Das **reale System** entwickelt sich nach unbekannter Dynamik:

$$s_{i+1}^r = \Phi_{\text{real}}(s_i^r,\, a_i^r;\, \theta^*)$$

Der **Simulator** approximiert diese Dynamik differenzierbar:

$$s_{i+1} = \Phi_{\text{sim}}(s_i^r,\, a_i^r;\, \theta)$$

wobei $s_i$ der Systemzustand $(q, \dot{q})$, $a_i^r$ die aufgezeichneten realen
Steuerbefehle und $\theta^*$ die unbekannten wahren Parameter sind.

Gegeben eine reale Trajektorie $\tau^r = \{s_0^r, a_0^r, \ldots, s_N^r\}$ ist
das Ziel, $\theta$ so zu schätzen, dass die simulierte Trajektorie
$\tau^s(\theta) = \{s_0^s, \ldots, s_N^s\}$ mit $s_0^s = s_0^r$ die reale
Trajektorie möglichst genau nachbildet.

### Loss-Funktion

Es wird ein **q-only MSE Loss** über zufällige Trajektorienfragmente verwendet.
Pro Iteration werden $B$ zufällige Startpunkte gezogen. Jedes Fragment der Länge
$H$ wird mit realem Anfangszustand initialisiert und mit den realen
Steuerbefehlen in MJX abgespielt:

$$\mathcal{L}(\theta) = \frac{1}{B \cdot H} \sum_{b=1}^{B} \sum_{i=0}^{H-1}
\left( q_{\text{sim},\,i+1}^{(b)}(\theta) - q_{\text{real},\,i+1}^{(b)} \right)^2$$

mit:
- $B$ — Batch-Größe (zufällige Startpunkte)
- $H$ — Horizont (Fragmentlänge in Schritten)
- $q_{\text{sim}}$ — simulierte Gelenkposition nach einem MJX-Schritt
- $q_{\text{real}}$ — aufgezeichnete reale Gelenkposition im nächsten Schritt

### Log-Parametrisierung

Da alle Parameter physikalisch positiv sein müssen, wird in Log-Space optimiert:

$$\varphi = \log(\theta), \quad \theta = \exp(\varphi)$$

Der Gradient $\frac{\partial \mathcal{L}}{\partial \varphi}$ wird mit
**Forward-Mode Automatic Differentiation** berechnet (`jax.jacfwd`), da
Reverse-Mode bei MJX-Constraints (interner `while_loop`) problematisch ist.

Die Optimierung erfolgt mit **Adam** auf $\varphi$.

### Gradienten: Forward-Mode vs. Reverse-Mode

MJX löst Constraints intern mit einem iterativen Solver (`while_loop`).
`jax.grad` (Reverse-Mode) kann durch solche Schleifen nicht stabil
differenzieren. `jax.jacfwd` (Forward-Mode) propagiert Tangenten vorwärts
und umgeht dieses Problem. Für wenige Parameter ($d = 3$) ist Forward-Mode
effizient genug.

## Datensatzformat

Die Skripte in diesem Ordner erwarten eine bereits konvertierte SysID-NPZ,
nicht die rohe ROS-NPZ.

Die Kangaroo-Datasets liegen auch lokal in diesem Ordner:

```text
scripts/Kangaroo/datasets/
```

Rohdaten zuerst konvertieren:

```bash
python scripts/Kangaroo/convert_kangaroo_real_npz.py \
  scripts/Kangaroo/datasets/DEINE_REAL_AUFNAHME.npz
```

Danach die erzeugte Datei verwenden:

```text
DEINE_REAL_AUFNAHME_sysid.npz
```

Die SysID-NPZ muss diese Keys enthalten:

```text
time
dt
ctrl
qpos
qvel
```

Erwartete Shapes:

```text
time: (T,)
dt:   scalar
ctrl: (T, 28)
qpos: (T, 39)
qvel: (T, 38)
```

Für `arm_left_4_joint` werden aktuell diese Indizes verwendet:

```text
ctrl[:, 19] -> Command
qpos[:, 12] -> Position q
qvel[:, 11] -> Geschwindigkeit dq
```

Nützliche Daten- und Plot-Skripte liegen ebenfalls in diesem Ordner:

```bash
# Rohe ROS-NPZ prüfen
python scripts/Kangaroo/plot_kangaroo_real_npz.py \
  scripts/Kangaroo/datasets/DEINE_REAL_AUFNAHME.npz

# ROS-NPZ in SysID-NPZ konvertieren
python scripts/Kangaroo/convert_kangaroo_real_npz.py \
  scripts/Kangaroo/datasets/DEINE_REAL_AUFNAHME.npz

# Konvertierte SysID-NPZ prüfen
python scripts/Kangaroo/plot_kangaroo_sysid_npz.py \
  scripts/Kangaroo/datasets/DEINE_REAL_AUFNAHME_sysid.npz

# Vor Optimierung: Sim-vs-Real plotten
python scripts/Kangaroo/plot_kangaroo_sim_vs_real.py \
  scripts/Kangaroo/datasets/DEINE_REAL_AUFNAHME_sysid.npz
```

## Aktueller Workflow

### 1. Rohdaten prüfen

```bash
python scripts/Kangaroo/plot_kangaroo_real_npz.py \
  scripts/Kangaroo/datasets/left_elbow_chirp_20260530_081011.npz
```

### 2. ROS-NPZ in SysID-NPZ konvertieren

```bash
python scripts/Kangaroo/convert_kangaroo_real_npz.py \
  scripts/Kangaroo/datasets/left_elbow_chirp_20260530_081011.npz
```

### 3. Konvertierte SysID-NPZ prüfen

```bash
python scripts/Kangaroo/plot_kangaroo_sysid_npz.py \
  scripts/Kangaroo/datasets/left_elbow_chirp_20260530_081011_sysid.npz
```

### 4. Vor Optimierung: MJX-vs-Real plotten

```bash
python scripts/Kangaroo/plot_kangaroo_sim_vs_real.py \
  scripts/Kangaroo/datasets/left_elbow_chirp_20260530_081011_sysid.npz
```

Der Plot wird automatisch in `scripts/Kangaroo/datasets/` gespeichert.

### 5. Parameter optimieren

`optimize_joint_params.py` nutzt Paper-nahe Trajektorien-Fragmente:

```text
pro Iteration:
  B zufällige Startpunkte aus der realen Trajektorie ziehen
  jedes Fragment mit qpos_real[start], qvel_real[start] initialisieren
  reale ctrl[start:start+horizon] in MJX abspielen
  q_sim_after_step mit q_real_next vergleichen
  q-only Loss über alle Fragmente mitteln
```

Zusätzlich wird ein fester Eval-Batch einmal am Anfang gezogen. Deshalb ist
`train` in der Ausgabe verrauscht, aber `eval` ist vergleichbar:

```text
train = aktueller zufälliger Batch
eval  = immer derselbe feste Eval-Batch
```

Alle drei aktuellen Parameter optimieren:

```bash
python scripts/Kangaroo/optimize_joint_params.py \
  --npz scripts/Kangaroo/datasets/left_elbow_chirp_20260530_081011_sysid.npz \
  --horizon 100 \
  --batch-size 16 \
  --eval-batch-size 16 \
  --max-iter 300 \
  --lr 0.002 \
  --armature 0.00663065 \
  --damping 0.0001 \
  --frictionloss 0.0001 \
  --optimize armature damping frictionloss
```

Wichtig: `damping` und `frictionloss` starten im XML bei `0.0`. Für die
Log-Parametrisierung ist ein kleiner positiver Startwert wie `0.0001`
praktischer.

### 6. Nach Optimierung plotten

Die besten Werte aus der Ausgabe in den Plot übernehmen:

```bash
python scripts/Kangaroo/plot_kangaroo_sim_vs_real.py \
  scripts/Kangaroo/datasets/left_elbow_chirp_20260530_081011_sysid.npz \
  --armature 0.00394267 \
  --damping 0.00014238 \
  --frictionloss 0.00010000 \
  --out scripts/Kangaroo/datasets/left_elbow_chirp_optimized_sim_vs_real.png
```

### 7. Video rendern

Nur MJX-Simulation:

```bash
python scripts/Kangaroo/render_dataset_video.py \
  --npz scripts/Kangaroo/datasets/left_elbow_chirp_20260530_081011_sysid.npz \
  --mode sim \
  --start 0 \
  --stop 12560 \
  --stride 4 \
  --fps 50
```

Real links, MJX rechts:

```bash
python scripts/Kangaroo/render_dataset_video.py \
  --npz scripts/Kangaroo/datasets/left_elbow_chirp_20260530_081011_sysid.npz \
  --mode side-by-side \
  --start 0 \
  --stop 12560 \
  --stride 4 \
  --fps 50
```

## Constraints, Gravity und Base

Für reale Kangaroo-Daten laufen die Skripte standardmäßig mit Gravity,
Equality Constraints und Contacts. Dafür muss kein Argument gesetzt werden.

Das ist für Kangaroo wichtig, weil das Modell mechanische Kopplungen bzw.
Closed-Loop-Strukturen enthält. Ohne diese Constraints sieht das Modell zwar
einfacher aus, aber körperlich falsch.

Nur für Debug-Tests können Constraints/Contacts deaktiviert werden:

```bash
--disable-constraints
```

Intern werden dann deaktiviert:

```python
mjDSBL_EQUALITY
mjDSBL_CONTACT
```

Gravity kann ebenfalls nur für Debug-Tests ausgeschaltet werden:

```bash
--no-gravity
```

Die Base ist standardmäßig gepinnt. Das heißt nach jedem Simulationsschritt
werden Floating-Base-Position, Orientierung und Geschwindigkeit wieder fixiert.
Mit `--free-base` kann dieses Verhalten ausgeschaltet werden.

Aktueller Standard für echte Arm-Daten:

```text
gravity:      on      (Default)
constraints:  on      (Default)
base:         pinned  (kein --free-base)
```

## Warum Forward-Mode?

MJX kann mit Constraints forward simulieren. Reverse-Mode Backpropagation
(`jax.value_and_grad`) ist bei Constraints aber problematisch, weil der interne
MJX-Solver einen `while_loop` verwenden kann. Deshalb nutzen die Kangaroo-Skripte
für Gradienten aktuell Forward-Mode:

```python
jax.jacfwd(loss_fn)
```

Das ist für wenige Parameter wie `armature`, `damping` und `frictionloss`
praktisch und robust genug.

## Aktuelle Skripte

```text
convert_kangaroo_real_npz.py
plot_kangaroo_real_npz.py
plot_kangaroo_sysid_npz.py
plot_kangaroo_sim_vs_real.py
```

Konvertieren und Prüfen der realen Kangaroo-Daten.

```text
optimize_joint_params.py
```

Optimiert ausgewählte Parameter (`armature`, `damping`, `frictionloss`) mit Adam
auf einem q-only Loss über zufällige Trajektorien-Fragmente.

```text
render_dataset_video.py
```

Rendert ein Video aus realem Dataset-Playback und/oder MJX-Rollout.

## Beispiel-Output

### Sim-vs-Real Plot

![MJX sim vs real](datasets/left_elbow_chirp_20260530_081011_sysid.sim_vs_real.png)

### MJX Animation

![MJX rollout](datasets/left_elbow_chirp_mjx_sim.gif)

[MP4 öffnen](datasets/left_elbow_chirp_mjx_sim.mp4)
