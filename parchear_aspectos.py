"""
parchear_aspectos.py
--------------------
Añade las columnas 'aspectos_positivos' y 'aspectos_negativos' al archivo
analisis_restaurantes.csv existente, sin reanalizar el resto de campos.

Solo llama a Gemini para los restaurantes que aún no tienen esas columnas,
por lo que es seguro interrumpir y reanudar.

Ejecucion: python parchear_aspectos.py
"""

import os
import re
import json
import time
import ast

import pandas as pd
from dotenv import load_dotenv
import google.generativeai as genai

# ── CONFIGURACION ─────────────────────────────────────────────────────────────

ARCHIVO_RESENAS  = "resenas_unificadas.csv"
ARCHIVO_ANALISIS = "analisis_restaurantes.csv"
MODELO_GEMINI    = "gemini-2.5-flash"
SLEEP_LLAMADAS   = 2
MAX_PALABRAS     = 500

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model_gemini = genai.GenerativeModel(MODELO_GEMINI)

# ── FUNCIONES ──────────────────────────────────────────────────────────────────

def extraer_aspectos(id_restaurante, resenas):
    texto = " | ".join(resenas)
    texto = " ".join(texto.split()[:MAX_PALABRAS])
    prompt = f"""Analiza estas reseñas de un restaurante y responde SOLO con JSON, sin texto extra ni backticks:
{{
  "aspectos_positivos": [],
  "aspectos_negativos": []
}}
Para "aspectos_positivos": lista de 3 a 5 puntos fuertes mencionados frecuentemente por los clientes (ej. "servicio muy amable", "raciones generosas").
Para "aspectos_negativos": lista de 1 a 3 quejas o puntos débiles recurrentes (ej. "tiempo de espera largo", "precio elevado"). Si no hay quejas claras, devuelve lista vacía.
Reseñas: {texto}"""

    for intento in range(3):
        try:
            response = model_gemini.generate_content(prompt)
            texto_r = re.sub(r'```json|```', '', response.text).strip()
            resultado = json.loads(texto_r)
            return resultado.get('aspectos_positivos', []), resultado.get('aspectos_negativos', [])
        except Exception as e:
            if '429' in str(e):
                espera = 10 * (intento + 1)
                print(f"  Cuota alcanzada, esperando {espera}s...")
                time.sleep(espera)
            elif '504' in str(e):
                time.sleep(5)
            else:
                print(f"  Error en restaurante {id_restaurante}: {e}")
                return None, None
    return None, None


def parsear_lista(v):
    if isinstance(v, list):
        return v
    try:
        return ast.literal_eval(v)
    except Exception:
        return []


# ── MAIN ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Cargar archivos
    print("Cargando archivos...")
    df_resenas  = pd.read_csv(ARCHIVO_RESENAS)
    df_analisis = pd.read_csv(ARCHIVO_ANALISIS)

    # Añadir columnas si no existen
    if 'aspectos_positivos' not in df_analisis.columns:
        df_analisis['aspectos_positivos'] = None
    if 'aspectos_negativos' not in df_analisis.columns:
        df_analisis['aspectos_negativos'] = None

    # Determinar qué restaurantes ya tienen los aspectos
    ya_procesados = df_analisis[
        df_analisis['aspectos_positivos'].notna() &
        df_analisis['aspectos_positivos'].astype(str).str.strip().ne('') &
        df_analisis['aspectos_positivos'].astype(str).str.strip().ne('[]') &
        df_analisis['aspectos_positivos'].astype(str).str.strip().ne('None')
    ]['Id_Restaurante'].astype(str).tolist()

    pendientes = df_analisis[
        ~df_analisis['Id_Restaurante'].astype(str).isin(ya_procesados)
    ]['Id_Restaurante'].tolist()

    print(f"Ya procesados: {len(ya_procesados)}")
    print(f"Pendientes:    {len(pendientes)}")

    if not pendientes:
        print("Todos los restaurantes ya tienen aspectos. Nada que hacer.")
    else:
        for i, id_rest in enumerate(pendientes, 1):
            resenas = df_resenas[
                df_resenas['Id_Restaurante'].astype(str) == str(id_rest)
            ]['Review'].dropna().tolist()

            if not resenas:
                print(f"  [{i}/{len(pendientes)}] Restaurante {id_rest}: sin reseñas, saltando.")
                continue

            print(f"  [{i}/{len(pendientes)}] Restaurante {id_rest}...")
            positivos, negativos = extraer_aspectos(id_rest, resenas)

            if positivos is None:
                print(f"    ✗ Fallo, se dejará vacío para reintentar la próxima vez.")
                continue

            mask = df_analisis['Id_Restaurante'].astype(str) == str(id_rest)
            df_analisis.loc[mask, 'aspectos_positivos'] = str(positivos)
            df_analisis.loc[mask, 'aspectos_negativos'] = str(negativos)

            # Guardar tras cada restaurante (checkpoint)
            df_analisis.to_csv(ARCHIVO_ANALISIS, index=False)
            time.sleep(SLEEP_LLAMADAS)

        print(f"\nListo. {ARCHIVO_ANALISIS} actualizado con aspectos_positivos y aspectos_negativos.")
