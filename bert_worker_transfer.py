# Una corrida de transferencia (T0-T3) sobre el corpus local; guarda los logits de validación de la mejor época.
import os, sys, json, time, random
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import tensorflow as tf
import tf_keras
from transformers import AutoTokenizer, TFAutoModel, TFAutoModelForSequenceClassification


def lotes(tok, txt, tam, max_len, orden=None):
    idx = np.arange(len(txt)) if orden is None else orden
    for i in range(0, len(idx), tam):
        b = idx[i:i + tam]
        enc = tok([txt[j] for j in b], padding=True, pad_to_multiple_of=16, truncation=True, max_length=max_len, return_tensors="tf")
        yield b, dict(enc)


def correr_t0(a, datos, tok, perdida):
    """Extractor congelado: promedio de los vectores de la última capa (BETO en modo inferencia) + cabeza lineal."""
    rutas = {k: f"{a['features']}_{k}.npy" for k in ("train", "val")}
    if not all(os.path.exists(r) for r in rutas.values()):
        bert = TFAutoModel.from_pretrained(a["checkpoint"])
        for k, txt in (("train", datos["train_text"]), ("val", datos["eval_text"])):
            fs = []
            for _, enc in lotes(tok, txt, 64, a["max_len"]):
                h = bert(enc, training=False).last_hidden_state.numpy()
                m = enc["attention_mask"].numpy()[..., None]
                fs.append((h * m).sum(1) / m.sum(1))
            np.save(rutas[k], np.concatenate(fs))
    Xtr, Xva = np.load(rutas["train"]), np.load(rutas["val"])
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr, Xva = (Xtr - mu) / sd, (Xva - mu) / sd
    y, ev_y = np.array(datos["train_y"]), np.array(datos["eval_y"])
    tf.random.set_seed(a["semilla"]); np.random.seed(a["semilla"])
    with tf.device("/CPU:0"):
        cabeza = tf_keras.Sequential([tf_keras.layers.Dense(2, input_shape=(Xtr.shape[1],))])
        opt = tf_keras.optimizers.legacy.Adam(learning_rate=a["lr_head"])
        hist = {"loss": [], "val_loss": []}; mejor, espera, mejor_lg, mejor_ep = 1e9, 0, None, 0
        rng = np.random.RandomState(a["semilla"]); tiempos = []
        for ep in range(a["epocas"]):
            orden = rng.permutation(len(y)); perd = []
            for i in range(0, len(y), 64):
                b = orden[i:i + 64]; t0 = time.perf_counter()
                with tf.GradientTape() as cinta:
                    l = perdida(y[b], cabeza(Xtr[b]))
                opt.apply_gradients(zip(cinta.gradient(l, cabeza.trainable_variables), cabeza.trainable_variables))
                perd.append(float(l) * len(b)); tiempos.append(time.perf_counter() - t0)
            lg = cabeza(Xva).numpy(); vl = float(perdida(ev_y, lg))
            hist["loss"].append(sum(perd) / len(y)); hist["val_loss"].append(vl)
            if vl < mejor - 1e-6:
                mejor, espera, mejor_lg, mejor_ep = vl, 0, lg, ep + 1
            else:
                espera += 1
                if espera >= a["paciencia"]:
                    break
    return hist, mejor_lg, mejor_ep, int(sum(np.prod(v.shape) for v in cabeza.trainable_variables)), float(np.median(tiempos))


def correr_ft(a, datos, tok, perdida):
    semilla = a["semilla"]
    textos, y = datos["train_text"], np.array(datos["train_y"])
    ev_txt, ev_y = datos["eval_text"], np.array(datos["eval_y"])
    modelo = TFAutoModelForSequenceClassification.from_pretrained(a["checkpoint"], num_labels=2)
    capas = modelo.bert.encoder.layer; n_capas = len(capas); k = a["n_desc"]
    modelo.bert.embeddings.trainable = (k == n_capas)
    for i, c in enumerate(capas):
        c.trainable = i >= n_capas - k
    vars_cab = [v for v in modelo.trainable_variables if "pooler" in v.name or "classifier" in v.name]
    vars_enc = [v for v in modelo.trainable_variables if not ("pooler" in v.name or "classifier" in v.name)]
    n_param = int(sum(np.prod(v.shape) for v in modelo.trainable_variables))

    n, lote, epocas = len(textos), a["lote"], a["epocas"]
    total = int(np.ceil(n / lote)) * epocas; pasos_warm = int(a["warmup"] * total)
    lr_enc, lr_cab = tf.Variable(a["lr_enc"], trainable=False), tf.Variable(a["lr_head"], trainable=False)
    opt_enc = tf_keras.optimizers.legacy.Adam(learning_rate=lr_enc, epsilon=1e-8)
    opt_cab = tf_keras.optimizers.legacy.Adam(learning_rate=lr_cab, epsilon=1e-8)

    def factor(paso):
        return paso / max(1, pasos_warm) if paso < pasos_warm else max(0.0, (total - paso) / max(1, total - pasos_warm))

    @tf.function(reduce_retracing=True)
    def paso_entrenamiento(entrada, etiquetas):
        with tf.GradientTape() as cinta:
            l = perdida(etiquetas, modelo(entrada, training=True).logits)
        todas = vars_enc + vars_cab
        grads, _ = tf.clip_by_global_norm(cinta.gradient(l, todas), a["recorte"])
        if vars_enc:
            opt_enc.apply_gradients(zip(grads[:len(vars_enc)], vars_enc))
        opt_cab.apply_gradients(zip(grads[len(vars_enc):], vars_cab))
        return l

    def predecir(txt):
        return np.concatenate([modelo(enc, training=False).logits.numpy() for _, enc in lotes(tok, txt, 64, a["max_len"])])

    hist = {"loss": [], "val_loss": []}; rng = np.random.RandomState(semilla); paso = 0
    mejor, espera, mejor_lg, mejor_ep, tiempos = 1e9, 0, None, 0, []
    for ep in range(epocas):
        orden = rng.permutation(n); perd = []
        for b, enc in lotes(tok, textos, lote, a["max_len"], orden):
            f = factor(paso); lr_enc.assign(a["lr_enc"] * f); lr_cab.assign(a["lr_head"] * f); paso += 1
            t0 = time.perf_counter()
            perd.append(float(paso_entrenamiento(enc, tf.constant(y[b]))) * len(b)); tiempos.append(time.perf_counter() - t0)
        lg = predecir(ev_txt); vl = float(perdida(ev_y, lg))
        hist["loss"].append(sum(perd) / n); hist["val_loss"].append(vl)
        with open(a["log"], "a") as f:
            f.write(f"{time.strftime('%H:%M:%S')} {a['etiqueta']} época {ep + 1}/{epocas} pérdida {hist['loss'][-1]:.4f} pérdida val {vl:.4f}\n")
        if vl < mejor - 1e-6:
            mejor, espera, mejor_lg, mejor_ep = vl, 0, lg, ep + 1
        else:
            espera += 1
            if espera >= a["paciencia"]:
                break
    return hist, mejor_lg, mejor_ep, n_param, float(np.median(tiempos))


if __name__ == "__main__":
    a = json.loads(sys.argv[1])
    os.environ["PYTHONHASHSEED"] = str(a["semilla"])
    random.seed(a["semilla"]); np.random.seed(a["semilla"]); tf.random.set_seed(a["semilla"])
    datos = json.load(open(a["datos"], encoding="utf-8"))
    tok = AutoTokenizer.from_pretrained(a["checkpoint"])
    perdida = tf_keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    t0 = time.time()
    f = correr_t0 if a["config"] == "T0" else correr_ft
    hist, lg, mejor_ep, n_param, paso_med = f(a, datos, tok, perdida)
    np.save(a["salida"] + "_val_logits.npy", lg)
    json.dump({"hist": hist, "mejor_epoca": mejor_ep, "parametros_entrenables": n_param,
               "paso_mediano_s": paso_med, "epocas_ejecutadas": len(hist["loss"])}, open(a["salida"] + ".json", "w"))
