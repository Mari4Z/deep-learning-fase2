# Corrida final (T0-T3): guarda historial por paso, gradientes, logits de validación y de prueba de la mejor época.
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


def correr_t0(a, datos, test, tok, perdida):
    """Extractor congelado: promedio de la última capa de BETO (modo inferencia) + cabeza lineal."""
    rutas = {k: f"{a['features']}_{k}.npy" for k in ("train", "val", "test")}
    textos = {"train": datos["train_text"], "val": datos["eval_text"], "test": test["test_text"]}
    if not all(os.path.exists(r) for r in rutas.values()):
        bert = TFAutoModel.from_pretrained(a["checkpoint"])
        for k, txt in textos.items():
            if os.path.exists(rutas[k]):
                continue
            fs = []
            for _, enc in lotes(tok, txt, 64, a["max_len"]):
                h = bert(enc, training=False).last_hidden_state.numpy(); m = enc["attention_mask"].numpy()[..., None]
                fs.append((h * m).sum(1) / m.sum(1))
            np.save(rutas[k], np.concatenate(fs))
    Xtr, Xva, Xte = (np.load(rutas[k]) for k in ("train", "val", "test"))
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr, Xva, Xte = (Xtr - mu) / sd, (Xva - mu) / sd, (Xte - mu) / sd
    y, ev_y = np.array(datos["train_y"]), np.array(datos["eval_y"])
    tf.random.set_seed(a["semilla"]); np.random.seed(a["semilla"])
    pasos = {"loss": [], "g_tot": [], "g_enc": [], "g_cab": []}
    with tf.device("/CPU:0"):
        cabeza = tf_keras.Sequential([tf_keras.layers.Dense(2, input_shape=(Xtr.shape[1],))])
        opt = tf_keras.optimizers.legacy.Adam(learning_rate=a["lr_head"])
        hist = {"loss": [], "val_loss": []}; curva = []; mejor, espera, mejor_ep = 1e9, 0, 0; lg_va = lg_te = None
        rng = np.random.RandomState(a["semilla"]); tiempos = []; paso = 0
        for ep in range(a["epocas"]):
            orden = rng.permutation(len(y)); perd = []
            for i in range(0, len(y), 64):
                b = orden[i:i + 64]; t0 = time.perf_counter()
                with tf.GradientTape() as cinta:
                    l = perdida(y[b], cabeza(Xtr[b]))
                gr = cinta.gradient(l, cabeza.trainable_variables)
                pasos["g_tot"].append(float(tf.linalg.global_norm(gr)))
                opt.apply_gradients(zip(gr, cabeza.trainable_variables)); paso += 1
                perd.append(float(l) * len(b)); pasos["loss"].append(float(l)); tiempos.append(time.perf_counter() - t0)
            lg = cabeza(Xva).numpy(); vl = float(perdida(ev_y, lg))
            hist["loss"].append(sum(perd) / len(y)); hist["val_loss"].append(vl); curva.append((paso, vl))
            if vl < mejor - 1e-6:
                mejor, espera, mejor_ep, lg_va, lg_te = vl, 0, ep + 1, lg, cabeza(Xte).numpy()
            else:
                espera += 1
                if espera >= a["paciencia"]:
                    break
    pasos["g_enc"] = [0.0] * len(pasos["loss"]); pasos["g_cab"] = pasos["g_tot"]
    return hist, curva, lg_va, lg_te, mejor_ep, int(sum(np.prod(v.shape) for v in cabeza.trainable_variables)), float(np.median(tiempos)), None, pasos


def correr_ft(a, datos, test, tok, perdida):
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
    pasos_ep = int(np.ceil(n / lote)); total = pasos_ep * epocas; pasos_warm = int(a["warmup"] * total)
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
        grads = cinta.gradient(l, todas)
        g_enc = tf.linalg.global_norm(grads[:len(vars_enc)]) if vars_enc else tf.constant(0.0)
        g_cab = tf.linalg.global_norm(grads[len(vars_enc):])
        g_tot = tf.linalg.global_norm(grads)
        grads, _ = tf.clip_by_global_norm(grads, a["recorte"])
        if vars_enc:
            opt_enc.apply_gradients(zip(grads[:len(vars_enc)], vars_enc))
        opt_cab.apply_gradients(zip(grads[len(vars_enc):], vars_cab))
        return l, g_tot, g_enc, g_cab

    t_inf = []
    def predecir(txt):
        out = []
        for _, enc in lotes(tok, txt, 64, a["max_len"]):
            t0 = time.perf_counter(); out.append(modelo(enc, training=False).logits.numpy()); t_inf.append((time.perf_counter() - t0, len(enc["input_ids"])))
        return np.concatenate(out)

    pasos = {"loss": [], "g_tot": [], "g_enc": [], "g_cab": []}
    hist = {"loss": [], "val_loss": []}; curva = []; rng = np.random.RandomState(a["semilla"]); paso = 0
    mejor, espera, mejor_ep, lg_va, lg_te, tiempos = 1e9, 0, 0, None, None, []
    mitad = pasos_ep // 2
    for ep in range(epocas):
        orden = rng.permutation(n); perd = []; i_ep = 0
        for b, enc in lotes(tok, textos, lote, a["max_len"], orden):
            f = factor(paso); lr_enc.assign(a["lr_enc"] * f); lr_cab.assign(a["lr_head"] * f); paso += 1; i_ep += 1
            t0 = time.perf_counter()
            l, gt, ge, gc = paso_entrenamiento(enc, tf.constant(y[b]))
            perd.append(float(l) * len(b)); tiempos.append(time.perf_counter() - t0)
            pasos["loss"].append(float(l)); pasos["g_tot"].append(float(gt)); pasos["g_enc"].append(float(ge)); pasos["g_cab"].append(float(gc))
            if i_ep == mitad:                       # diagnóstico a media época (no interviene en la selección)
                curva.append((paso, float(perdida(ev_y, predecir(ev_txt)))))
        lg = predecir(ev_txt); vl = float(perdida(ev_y, lg))
        hist["loss"].append(sum(perd) / n); hist["val_loss"].append(vl); curva.append((paso, vl))
        with open(a["log"], "a") as f:
            f.write(f"{time.strftime('%H:%M:%S')} {a['etiqueta']} época {ep + 1}/{epocas} pérdida {hist['loss'][-1]:.4f} pérdida val {vl:.4f}\n")
        if vl < mejor - 1e-6:
            mejor, espera, mejor_ep, lg_va = vl, 0, ep + 1, lg
            lg_te = predecir(test["test_text"])      # solo inferencia; se guarda la de la mejor época por validación
        else:
            espera += 1
            if espera >= a["paciencia"]:
                break
    med = float(np.median([t / m for t, m in t_inf if m == 64]))      # s por tuit en lotes completos
    return hist, curva, lg_va, lg_te, mejor_ep, n_param, float(np.median(tiempos)), med * 1000, pasos


if __name__ == "__main__":
    a = json.loads(sys.argv[1])
    os.environ["PYTHONHASHSEED"] = str(a["semilla"])
    random.seed(a["semilla"]); np.random.seed(a["semilla"]); tf.random.set_seed(a["semilla"])
    datos = json.load(open(a["datos"], encoding="utf-8")); test = json.load(open(a["datos_test"], encoding="utf-8"))
    tok = AutoTokenizer.from_pretrained(a["checkpoint"])
    perdida = tf_keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    f = correr_t0 if a["config"] == "T0" else correr_ft
    hist, curva, lg_va, lg_te, mejor_ep, n_param, paso_med, inf_1000, pasos = f(a, datos, test, tok, perdida)
    np.save(a["salida"] + "_val_logits.npy", lg_va); np.save(a["salida"] + "_test_logits.npy", lg_te)
    np.savez(a["salida"] + "_pasos.npz", **{k: np.array(v) for k, v in pasos.items()})
    json.dump({"hist": hist, "curva_val": curva, "mejor_epoca": mejor_ep, "parametros_entrenables": n_param, "paso_mediano_s": paso_med,
               "inferencia_s_por_1000": inf_1000, "epocas_ejecutadas": len(hist["loss"]), "pasos_totales": len(pasos["loss"])}, open(a["salida"] + ".json", "w"))
