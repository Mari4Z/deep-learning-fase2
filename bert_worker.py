# Proceso independiente que entrena UNA corrida de BERT y guarda logits e historial en disco.
import os, sys, json, time, random
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import tensorflow as tf
import tf_keras
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
from sklearn.model_selection import train_test_split


def entrenar(a):
    semilla = a["semilla"]
    os.environ["PYTHONHASHSEED"] = str(semilla)
    random.seed(semilla); np.random.seed(semilla); tf.random.set_seed(semilla)

    datos = json.load(open(a["datos"], encoding="utf-8"))
    textos, y = datos["train_text"], np.array(datos["train_y"])
    tok = AutoTokenizer.from_pretrained(a["checkpoint"])
    try:
        modelo = TFAutoModelForSequenceClassification.from_pretrained(a["checkpoint"], num_labels=2)
    except Exception:
        modelo = TFAutoModelForSequenceClassification.from_pretrained(a["checkpoint"], num_labels=2, from_pt=True)

    if a["validacion"]:      # 15 % del entrenamiento solo para curvas de diagnóstico
        textos, t_va, y, y_va = train_test_split(textos, y, test_size=0.15, random_state=semilla, stratify=y)
    n, lote, epocas = len(textos), a["lote"], a["epocas"]
    pasos_ep = int(np.ceil(n / lote)); total = pasos_ep * epocas
    pasos_warm = int(a["warmup"] * total) if a["planificacion"] else 0
    lr = a["lr"]

    lr_var = tf.Variable(lr, trainable=False, dtype=tf.float32)
    opt = tf_keras.optimizers.legacy.Adam(learning_rate=lr_var, epsilon=1e-8)
    perdida = tf_keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    def tasa(paso):
        if not a["planificacion"]:
            return lr
        if paso < pasos_warm:
            return lr * paso / max(1, pasos_warm)
        return lr * max(0.0, (total - paso) / max(1, total - pasos_warm))

    @tf.function(reduce_retracing=True)
    def paso_entrenamiento(entrada, etiquetas):
        with tf.GradientTape() as cinta:
            logits = modelo(entrada, training=True).logits
            l = perdida(etiquetas, logits)
        grads = cinta.gradient(l, modelo.trainable_variables)
        if a["planificacion"] and a["recorte"]:
            grads, _ = tf.clip_by_global_norm(grads, a["recorte"])
        opt.apply_gradients(zip(grads, modelo.trainable_variables))
        return l, logits

    def lotes(txt, tam, orden=None):
        idx = np.arange(len(txt)) if orden is None else orden
        for i in range(0, len(idx), tam):
            b = idx[i:i + tam]
            # relleno en múltiplos de 16: pocas formas distintas de entrada (evita recompilar el grafo en cada lote)
            enc = tok([txt[j] for j in b], padding=True, pad_to_multiple_of=16, truncation=True,
                      max_length=a["max_len"], return_tensors="tf")
            yield b, dict(enc)

    def predecir(txt):
        return np.concatenate([modelo(enc, training=False).logits.numpy() for _, enc in lotes(txt, 64)])

    hist = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}
    rng = np.random.RandomState(semilla); paso = 0; t0 = time.time()
    for ep in range(epocas):
        orden = rng.permutation(n); perd, acierto = [], 0
        for b, enc in lotes(textos, lote, orden):
            lr_var.assign(tasa(paso)); paso += 1
            l, logits = paso_entrenamiento(enc, tf.constant(y[b]))
            perd.append(float(l) * len(b)); acierto += int((logits.numpy().argmax(1) == y[b]).sum())
        hist["loss"].append(sum(perd) / n); hist["accuracy"].append(acierto / n)
        if a["validacion"]:
            lg = predecir(t_va)
            hist["val_loss"].append(float(perdida(y_va, lg))); hist["val_accuracy"].append(float((lg.argmax(1) == y_va).mean()))
        with open(a["log"], "a") as f:
            f.write(f"{time.strftime('%H:%M:%S')} {a['etiqueta']} época {ep + 1}/{epocas} "
                    f"pérdida {hist['loss'][-1]:.4f} acc {hist['accuracy'][-1]:.4f}\n")
    logits_te = predecir(datos["test_text"])
    return hist, logits_te, time.time() - t0


if __name__ == "__main__":
    a = json.loads(sys.argv[1])
    hist, logits, seg = entrenar(a)
    np.save(a["salida"] + "_logits.npy", logits)
    json.dump({"hist": hist, "segundos": seg}, open(a["salida"] + ".json", "w"))
