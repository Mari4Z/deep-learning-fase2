# Entrena BETO sobre el corpus local y guarda, por época, los logits de validación.
import os, sys, json, time, random
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import tensorflow as tf
import tf_keras
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification


def entrenar(a):
    semilla = a["semilla"]
    os.environ["PYTHONHASHSEED"] = str(semilla)
    random.seed(semilla); np.random.seed(semilla); tf.random.set_seed(semilla)
    datos = json.load(open(a["datos"], encoding="utf-8"))
    textos, y = datos["train_text"], np.array(datos["train_y"])
    ev_txt, ev_y = datos["eval_text"], np.array(datos["eval_y"])

    tok = AutoTokenizer.from_pretrained(a["checkpoint"])
    modelo = TFAutoModelForSequenceClassification.from_pretrained(a["checkpoint"], num_labels=2)
    n, lote, epocas = len(textos), a["lote"], a["epocas"]
    total = int(np.ceil(n / lote)) * epocas
    pasos_warm = int(a["warmup"] * total)
    lr = a["lr"]
    lr_var = tf.Variable(lr, trainable=False, dtype=tf.float32)
    opt = tf_keras.optimizers.legacy.Adam(learning_rate=lr_var, epsilon=1e-8)
    perdida = tf_keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    def tasa(paso):
        if paso < pasos_warm:
            return lr * paso / max(1, pasos_warm)
        return lr * max(0.0, (total - paso) / max(1, total - pasos_warm))

    @tf.function(reduce_retracing=True)
    def paso_entrenamiento(entrada, etiquetas):
        with tf.GradientTape() as cinta:
            logits = modelo(entrada, training=True).logits
            l = perdida(etiquetas, logits)
        grads = cinta.gradient(l, modelo.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, a["recorte"])
        opt.apply_gradients(zip(grads, modelo.trainable_variables))
        return l

    def lotes(txt, tam, orden=None):
        idx = np.arange(len(txt)) if orden is None else orden
        for i in range(0, len(idx), tam):
            b = idx[i:i + tam]
            enc = tok([txt[j] for j in b], padding=True, pad_to_multiple_of=16, truncation=True,
                      max_length=a["max_len"], return_tensors="tf")
            yield b, dict(enc)

    def predecir(txt):
        return np.concatenate([modelo(enc, training=False).logits.numpy() for _, enc in lotes(txt, 64)])

    hist = {"loss": [], "val_loss": []}
    rng = np.random.RandomState(semilla); paso = 0; t0 = time.time()
    for ep in range(epocas):
        orden = rng.permutation(n); perd = []
        for b, enc in lotes(textos, lote, orden):
            lr_var.assign(tasa(paso)); paso += 1
            perd.append(float(paso_entrenamiento(enc, tf.constant(y[b]))) * len(b))
        lg = predecir(ev_txt)
        np.save(f"{a['salida']}_val_logits_ep{ep + 1}.npy", lg)
        hist["loss"].append(sum(perd) / n)
        hist["val_loss"].append(float(perdida(ev_y, lg)))
        with open(a["log"], "a") as f:
            f.write(f"{time.strftime('%H:%M:%S')} local {a['etiqueta']} época {ep + 1}/{epocas} "
                    f"pérdida {hist['loss'][-1]:.4f} pérdida val {hist['val_loss'][-1]:.4f}\n")
    return hist, time.time() - t0


if __name__ == "__main__":
    a = json.loads(sys.argv[1])
    hist, seg = entrenar(a)
    json.dump({"hist": hist, "segundos": seg}, open(a["salida"] + ".json", "w"))
