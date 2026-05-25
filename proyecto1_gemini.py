"""
proyecto1_gemini.py
-------------------
Sistema de recomendacion de restaurantes de Madrid usando Gemini + ChromaDB + LangGraph.

Genera analisis_restaurantes.csv con criterios booleanos extraidos por Gemini.
Construye un agente conversacional con cinco herramientas:
  - filtrar_por_criterios: filtrado booleano
  - buscar_y_razonar: ChromaDB + RAG principal
  - buscar_por_plato: busqueda por plato especifico
  - buscar_por_nombre: busqueda por nombre de restaurante
  - buscar_por_cercania: busqueda por proximidad geografica

Ejecucion: python proyecto1_gemini.py
"""

import os
import re
import json
import time
import glob
import ast

import pandas as pd
from dotenv import load_dotenv
import google.generativeai as genai
import chromadb
from chromadb.utils import embedding_functions
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent

# ── CONFIGURACION ─────────────────────────────────────────────────────────────

CARPETA_RESENAS  = "Resenas_TA"
ARCHIVO_RANKING  = "ta_calles.csv"
ARCHIVO_RESENAS  = "resenas_unificadas.csv"
ARCHIVO_ANALISIS = "analisis_restaurantes.csv"
CARPETA_CHROMA   = "chromadb"
ARCHIVO_GEO      = "restaurantes_geo.csv"
MODELO_GEMINI    = "gemini-2.5-flash"
SLEEP_LLAMADAS   = 2
MAX_PALABRAS     = 500

CRITERIOS_BOOLEANOS = [
    "buena_comida", "buen_servicio", "buen_ambiente",
    "buena_relacion_precio_calidad", "espera_corta",
    "apto_ninos", "apto_mascotas", "cocina_tradicional",
    "cocina_moderna", "comida_sencilla", "comida_elaborada"
]

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model_gemini = genai.GenerativeModel(MODELO_GEMINI)

# ── PASO 1: UNIFICAR Y LIMPIAR RESENAS ────────────────────────────────────────

def limpiar_resena(texto):
    if not isinstance(texto, str):
        return ""
    texto = texto.lower()
    texto = re.sub(r'http\S+|www\S+', '', texto)
    texto = re.sub(r'[^\w\sáéíóúüñ]', ' ', texto)
    texto = re.sub(r'\b\d+\b', '', texto)
    texto = re.sub(r'\s+', ' ', texto).strip()
    return texto

def unificar_resenas():
    print("Cargando resenas...")
    archivos = sorted(glob.glob(f"{CARPETA_RESENAS}/*.csv"))
    if not archivos:
        raise FileNotFoundError(f"No se encontraron CSVs en: {CARPETA_RESENAS}")
    dfs = []
    for i, archivo in enumerate(archivos, start=1):
        df_temp = pd.read_csv(archivo)
        df_temp = df_temp[['web_scraper_order', 'data3']]
        df_temp = df_temp.rename(columns={'web_scraper_order': 'Id_review', 'data3': 'Review'})
        df_temp['Id_Restaurante'] = i
        df_temp['Id_review'] = range(1, len(df_temp) + 1)
        dfs.append(df_temp)
    df = pd.concat(dfs, ignore_index=True)
    df = df[['Id_Restaurante', 'Id_review', 'Review']]
    df['Review'] = df['Review'].apply(limpiar_resena)
    print(f"  {len(df):,} resenas de {len(archivos)} restaurantes")
    return df

# ── PASO 2: ANALIZAR CON GEMINI ───────────────────────────────────────────────

def analizar_restaurante(id_restaurante, resenas):
    texto = " | ".join(resenas)
    texto = " ".join(texto.split()[:MAX_PALABRAS])
    prompt = f"""Analiza estas resenas y responde SOLO con JSON, sin texto extra:
{{
  "buena_comida": bool,
  "buen_servicio": bool,
  "buen_ambiente": bool,
  "buena_relacion_precio_calidad": bool,
  "espera_corta": bool,
  "apto_ninos": bool,
  "apto_mascotas": bool,
  "cocina_tradicional": bool,
  "cocina_moderna": bool,
  "comida_sencilla": bool,
  "comida_elaborada": bool,
  "platos_destacados": [],
  "aspectos_positivos": [],
  "aspectos_negativos": [],
  "resumen": ""
}}
Para "aspectos_positivos" incluye una lista de 3 a 5 puntos fuertes mencionados frecuentemente por los clientes (ej. "servicio muy amable", "raciones generosas").
Para "aspectos_negativos" incluye una lista de 1 a 3 puntos debiles o quejas recurrentes (ej. "tiempo de espera largo", "precio elevado"). Si no hay quejas claras, devuelve lista vacia.
Resenas: {texto}"""
    for intento in range(3):
        try:
            response = model_gemini.generate_content(prompt)
            texto_r = re.sub(r'```json|```', '', response.text).strip()
            resultado = json.loads(texto_r)
            resultado['Id_Restaurante'] = str(id_restaurante)
            return resultado
        except Exception as e:
            if '429' in str(e):
                espera = 10 * (intento + 1)
                print(f"  Cuota alcanzada, esperando {espera}s...")
                time.sleep(espera)
            elif '504' in str(e):
                time.sleep(5)
            else:
                return {'Id_Restaurante': str(id_restaurante), 'error': str(e)}
    return {'Id_Restaurante': str(id_restaurante), 'error': 'max reintentos'}

def analizar_todos(df_completo):
    if os.path.exists(ARCHIVO_ANALISIS):
        df_prev = pd.read_csv(ARCHIVO_ANALISIS)
        df_ok = df_prev[df_prev['error'].isna()] if 'error' in df_prev.columns else df_prev
        ids_ok = set(df_ok['Id_Restaurante'].astype(str).tolist())
        resultados = df_ok.to_dict('records')
        print(f"Retomando: {len(ids_ok)} ya procesados")
    else:
        ids_ok = set()
        resultados = []

    pendientes = [r for r in df_completo['Id_Restaurante'].unique() if str(r) not in ids_ok]
    print(f"Pendientes: {len(pendientes)}")
    for i, id_rest in enumerate(pendientes, 1):
        resenas = df_completo[df_completo['Id_Restaurante'] == id_rest]['Review'].dropna().tolist()
        print(f"  [{i}/{len(pendientes)}] Restaurante {id_rest}...")
        resultados.append(analizar_restaurante(id_rest, resenas))
        pd.DataFrame(resultados).to_csv(ARCHIVO_ANALISIS, index=False)
        time.sleep(SLEEP_LLAMADAS)
    return pd.DataFrame(resultados)

# ── PASO 3: CONSTRUIR CHROMADB ────────────────────────────────────────────────

def construir_chromadb(df):
    print("Construyendo ChromaDB...")
    cliente = chromadb.PersistentClient(path=CARPETA_CHROMA)
    try:
        cliente.delete_collection("restaurantes")
    except Exception:
        pass
    ef = embedding_functions.DefaultEmbeddingFunction()
    coleccion = cliente.create_collection("restaurantes", embedding_function=ef)
    documentos, metadatos, ids = [], [], []
    for _, fila in df.iterrows():
        platos = ", ".join(fila['platos_destacados']) if isinstance(fila['platos_destacados'], list) else ""
        pos = ", ".join(fila['aspectos_positivos']) if isinstance(fila.get('aspectos_positivos'), list) else ""
        neg = ", ".join(fila['aspectos_negativos']) if isinstance(fila.get('aspectos_negativos'), list) else ""
        texto = f"{fila['resumen']} Platos destacados: {platos} Positivos: {pos} Negativos: {neg}"
        meta = {k: (bool(fila[k]) if k in CRITERIOS_BOOLEANOS else str(fila[k])) for k in
                ['Id_Restaurante', 'Restaurante', 'Dirección', 'platos_destacados'] + CRITERIOS_BOOLEANOS}
        meta['Valoracion'] = float(fila['Valoracion'])
        meta['Votaciones'] = int(fila['Votaciones'])
        meta['Dirección'] = str(fila['Dirección'])
        documentos.append(texto)
        metadatos.append(meta)
        ids.append(str(fila['Id_Restaurante']))
    coleccion.add(documents=documentos, metadatas=metadatos, ids=ids)
    print(f"  {coleccion.count()} restaurantes indexados")
    return coleccion

# ── PASO 3.5: GEOCODIFICAR RESTAURANTES ──────────────────────────────────────

def geocodificar_restaurantes(df):
    # Si ya existe el archivo geo con coordenadas, cargarlo y hacer merge
    if os.path.exists(ARCHIVO_GEO):
        df_geo = pd.read_csv(ARCHIVO_GEO)[['Id_Restaurante', 'latitud', 'longitud']]
        df_geo['Id_Restaurante'] = df_geo['Id_Restaurante'].astype(str)
        df = df.merge(df_geo, on='Id_Restaurante', how='left')
        if df['latitud'].notna().sum() > 0:
            print(f"Coordenadas cargadas desde {ARCHIVO_GEO}.")
            return df

    print("Geocodificando direcciones (puede tardar ~90 segundos)...")
    geolocator = Nominatim(user_agent="restaurantes_madrid")
    latitudes, longitudes = [], []
    for i, (_, fila) in enumerate(df.iterrows(), 1):
        try:
            direccion = f"{fila['Dirección']}, Madrid, España"
            loc = geolocator.geocode(direccion, timeout=10)
            if loc:
                latitudes.append(loc.latitude)
                longitudes.append(loc.longitude)
            else:
                latitudes.append(None)
                longitudes.append(None)
        except Exception:
            latitudes.append(None)
            longitudes.append(None)
        print(f"  [{i}/{len(df)}] geocodificando...")
        time.sleep(1)  # Nominatim exige 1s entre llamadas
    df['latitud'] = latitudes
    df['longitud'] = longitudes
    # Guardar solo coordenadas en archivo separado para no corromper analisis
    df[['Id_Restaurante', 'latitud', 'longitud']].to_csv(ARCHIVO_GEO, index=False)
    print(f"Geocodificacion completada: {sum(l is not None for l in latitudes)}/{len(df)} exitosas.")
    return df

# ── TOOLS Y AGENTE ────────────────────────────────────────────────────────────

df_global = None
coleccion_global = None

@tool
def filtrar_por_criterios(criterios: str) -> str:
    """Filtra restaurantes por criterios booleanos. Recibe un JSON,
    ejemplo: {"apto_ninos": true, "cocina_tradicional": true}"""
    try:
        filtros = json.loads(criterios)
    except Exception:
        return "Error: JSON invalido."
    df_f = df_global.copy()
    for k, v in filtros.items():
        if k in CRITERIOS_BOOLEANOS:
            df_f = df_f[df_f[k] == v]
    if df_f.empty:
        return "No se encontraron restaurantes."
    df_f = df_f.sort_values(['Valoracion', 'Votaciones'], ascending=False).head(5)
    return "\n\n".join([
        f"Restaurante: {r['Restaurante']} | Valoracion: {r['Valoracion']} ({r['Votaciones']} votos)\n"
        f"Direccion: {r['Dirección']}\nResumen: {r['resumen']}\n"
        f"Platos: {', '.join(r['platos_destacados']) if isinstance(r['platos_destacados'], list) else r['platos_destacados']}\n"
        f"Positivos: {chr(10).join('- '+p for p in r['aspectos_positivos']) if isinstance(r.get('aspectos_positivos'), list) else ''}\n"
        f"Negativos: {chr(10).join('- '+n for n in r['aspectos_negativos']) if isinstance(r.get('aspectos_negativos'), list) and r['aspectos_negativos'] else 'Ninguno destacable'}"
        for _, r in df_f.iterrows()
    ])

@tool
def buscar_y_razonar(consulta: str) -> str:
    """Herramienta principal RAG. Recupera los restaurantes mas relevantes
    combinando busqueda semantica con ChromaDB y filtros booleanos si aplican,
    y devuelve contexto completo para que Gemini razone la mejor respuesta.
    Usar para CUALQUIER tipo de consulta."""

    # 1. Busqueda semantica en ChromaDB — recuperar candidatos
    res = coleccion_global.query(query_texts=[consulta], n_results=15)
    ids_semanticos = [m['Id_Restaurante'] for m in res['metadatas'][0]]

    # 2. Candidatos del pool semantico
    df_candidatos = df_global[df_global['Id_Restaurante'].astype(str).isin(ids_semanticos)].copy()

    # 3. Aplicar filtros booleanos si la consulta los menciona explicitamente
    filtros_detectados = {
        'apto_ninos'                  : any(w in consulta.lower() for w in ['nino', 'niño', 'familia', 'hijo', 'peque']),
        'apto_mascotas'               : any(w in consulta.lower() for w in ['perro', 'mascota', 'animal']),
        'buena_relacion_precio_calidad': any(w in consulta.lower() for w in ['precio', 'economico', 'barato', 'calidad precio']),
        'espera_corta'                : any(w in consulta.lower() for w in ['rapido', 'sin espera', 'espera corta']),
        'cocina_tradicional'          : any(w in consulta.lower() for w in ['tradicional', 'clasico', 'tipico']),
        'cocina_moderna'              : any(w in consulta.lower() for w in ['moderno', 'innovador', 'fusion', 'creativo']),
    }

    for criterio, detectado in filtros_detectados.items():
        if detectado and criterio in df_candidatos.columns:
            df_filtrado = df_candidatos[df_candidatos[criterio] == True]
            if not df_filtrado.empty:
                df_candidatos = df_filtrado

    # 4. Ordenar por valoracion y tomar top 8
    df_candidatos = df_candidatos.sort_values(['Valoracion', 'Votaciones'], ascending=False).head(8)

    if df_candidatos.empty:
        # Si no hay candidatos con filtros, devolver los mejores semanticos sin filtrar
        df_candidatos = df_global[df_global['Id_Restaurante'].astype(str).isin(ids_semanticos)]
        df_candidatos = df_candidatos.sort_values(['Valoracion', 'Votaciones'], ascending=False).head(8)

    # 5. Construir contexto rico para Gemini
    contexto = []
    for _, r in df_candidatos.iterrows():
        platos = ', '.join(r['platos_destacados']) if isinstance(r['platos_destacados'], list) else str(r['platos_destacados'])
        contexto.append(
            f"[{r['Restaurante']}]\n"
            f"Valoracion: {r['Valoracion']} | Votos: {r['Votaciones']}\n"
            f"Direccion: {r['Dirección']}\n"
            f"Resumen: {r['resumen']}\n"
            f"Platos destacados: {platos}\n"
            f"Apto ninos: {r['apto_ninos']} | Mascotas: {r['apto_mascotas']} | "
            f"Tradicional: {r['cocina_tradicional']} | Moderno: {r['cocina_moderna']}\n"
            f"Aspectos positivos: {chr(10).join('- '+p for p in r['aspectos_positivos']) if isinstance(r.get('aspectos_positivos'), list) else ''}\n"
            f"Aspectos negativos: {chr(10).join('- '+n for n in r['aspectos_negativos']) if isinstance(r.get('aspectos_negativos'), list) and r['aspectos_negativos'] else 'Ninguno destacable'}"
        )
    return "\n\n---\n\n".join(contexto)


@tool
def buscar_por_plato(plato: str) -> str:
    """Busca restaurantes que tengan un plato concreto entre sus destacados.
    Ejemplo: 'croquetas', 'pulpo', 'cocido'."""
    plato_l = plato.lower().strip()
    df_f = df_global[df_global['platos_destacados'].apply(
        lambda lista: isinstance(lista, list) and any(plato_l in p.lower() for p in lista)
    )]
    if df_f.empty:
        return f"No se encontraron restaurantes con '{plato}'."
    df_f = df_f.sort_values(['Valoracion', 'Votaciones'], ascending=False).head(5)
    return "\n\n".join([
        f"Restaurante: {r['Restaurante']} | Valoracion: {r['Valoracion']} ({r['Votaciones']} votos)\n"
        f"Direccion: {r['Dirección']}\nPlatos: {', '.join(r['platos_destacados'])}"
        for _, r in df_f.iterrows()
    ])


@tool
def buscar_por_nombre(nombre: str) -> str:
    """Busca un restaurante por su nombre exacto o aproximado.
    Usar siempre que el usuario pregunte por un restaurante concreto por su nombre."""
    nombre_l = nombre.lower().strip()
    df_f = df_global[df_global['Restaurante'].str.lower().str.contains(nombre_l, na=False)]
    if df_f.empty:
        return f"No se encontró ningún restaurante con el nombre '{nombre}'."
    df_f = df_f.sort_values(['Valoracion', 'Votaciones'], ascending=False)
    return "\n\n".join([
        f"Restaurante: {r['Restaurante']} | Valoracion: {r['Valoracion']} ({r['Votaciones']} votos)\n"
        f"Direccion: {r['Dirección']}\nResumen: {r['resumen']}\n"
        f"Platos: {', '.join(r['platos_destacados']) if isinstance(r['platos_destacados'], list) else r['platos_destacados']}\n"
        f"Positivos: {chr(10).join('- '+p for p in r['aspectos_positivos']) if isinstance(r.get('aspectos_positivos'), list) else ''}\n"
        f"Negativos: {chr(10).join('- '+n for n in r['aspectos_negativos']) if isinstance(r.get('aspectos_negativos'), list) and r['aspectos_negativos'] else 'Ninguno destacable'}"
        for _, r in df_f.iterrows()
    ])


@tool
def buscar_por_cercania(ubicacion: str) -> str:
    """Busca los restaurantes mas cercanos a una ubicacion o coordenadas.
    El usuario puede dar su ubicacion como direccion, barrio, o coordenadas 'lat,lon'.
    Ejemplo: 'Puerta del Sol', 'Malasana', '40.4168,-3.7038'."""
    # Obtener coordenadas del usuario
    try:
        partes = ubicacion.split(',')
        if len(partes) == 2 and all(p.strip().replace('.', '').replace('-', '').isdigit()
                                     for p in partes):
            lat, lon = map(float, partes)
        else:
            geolocator = Nominatim(user_agent="restaurantes_madrid")
            loc = geolocator.geocode(f"{ubicacion}, Madrid, España", timeout=10)
            if not loc:
                return f"No se pudo encontrar la ubicacion '{ubicacion}'."
            lat, lon = loc.latitude, loc.longitude
    except Exception as e:
        return f"Error al obtener ubicacion: {e}"

    # Calcular distancias solo para restaurantes geocodificados
    df_geo = df_global[df_global['latitud'].notna() & df_global['longitud'].notna()].copy()
    if df_geo.empty:
        return "No hay coordenadas disponibles. Ejecuta el script de nuevo para geocodificar."

    df_geo['distancia_km'] = df_geo.apply(
        lambda r: geodesic((lat, lon), (r['latitud'], r['longitud'])).km, axis=1
    )
    df_cercanos = df_geo.sort_values('distancia_km').head(5)

    return "\n\n".join([
        f"Restaurante: {r['Restaurante']} | {r['distancia_km']:.2f} km\n"
        f"Valoracion: {r['Valoracion']} ({r['Votaciones']} votos)\n"
        f"Direccion: {r['Dirección']}\nResumen: {r['resumen']}"
        for _, r in df_cercanos.iterrows()
    ])


def construir_agente(df, coleccion):
    global df_global, coleccion_global
    df_global = df
    coleccion_global = coleccion
    llm = ChatGoogleGenerativeAI(model=MODELO_GEMINI, google_api_key=os.getenv("GEMINI_API_KEY"))
    tools = [buscar_y_razonar, filtrar_por_criterios, buscar_por_plato, buscar_por_nombre, buscar_por_cercania]
    prompt = """Eres un asistente experto en recomendacion de restaurantes de Madrid.

Tienes cinco herramientas:
1. buscar_y_razonar: herramienta principal RAG. Usala para CUALQUIER consulta general,
   tipos de cocina (india, italiana, peruana, japonesa...), experiencias (romantico,
   familiar, celebracion), ambiente, precio, etc. Te devuelve contexto completo
   sobre los restaurantes mas relevantes para que tu razones la mejor respuesta.
2. filtrar_por_criterios: solo para filtros booleanos muy especificos cuando el usuario
   pide explicitamente criterios como apto_ninos, apto_mascotas, etc. sin contexto adicional.
3. buscar_por_plato: cuando el usuario menciona un plato concreto.
4. buscar_por_nombre: cuando el usuario pregunta por un restaurante concreto por su nombre.
   Usarla SIEMPRE que el usuario mencione el nombre de un restaurante especifico.
5. buscar_por_cercania: cuando el usuario quiere restaurantes cerca de un sitio o de donde esta.
   Si el usuario no da su ubicacion, preguntasela antes de llamar a la herramienta.

Cuando recomiendes restaurantes SIEMPRE estructura tu respuesta así para cada uno:
- Nombre y valoración
- Por qué encaja con lo que pide el usuario
- ✅ Aspectos positivos destacados por los clientes (usa los aspectos_positivos del contexto)
- ⚠️ Aspectos a tener en cuenta (usa los aspectos_negativos del contexto; si no hay, dilo)
- Platos recomendados

Es OBLIGATORIO mencionar los aspectos negativos cuando existan. No los omitas aunque sean menores.
Si el contexto no incluye restaurantes del tipo pedido, dilo honestamente.

Responde en espanol de forma cercana y natural."""
    return create_react_agent(llm, tools, prompt=prompt)

def recomendar(agente, consulta, historial):
    historial.append({"role": "user", "content": consulta})
    respuesta = agente.invoke({"messages": historial})
    ultimo = respuesta["messages"][-1].content
    historial.append({"role": "assistant", "content": ultimo})
    return ultimo

# ── MAIN ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    def parsear_platos(v):
        if isinstance(v, list): return v
        try: return ast.literal_eval(v)
        except: return []

    # Siempre regenerar resenas para capturar nuevos restaurantes
    df_resenas = unificar_resenas()
    df_ranking = pd.read_csv(ARCHIVO_RANKING, sep=";", skiprows=1)
    df_ranking = df_ranking[['Id_Restaurante', 'Restaurante', 'Votaciones', 'Valoracion', 'Dirección']]
    # LEFT merge: los restaurantes sin entrada en ta_calles no se pierden
    df_completo = pd.merge(df_resenas, df_ranking, on='Id_Restaurante', how='left')
    # Rellenar metadatos ausentes con valores por defecto
    df_completo['Restaurante'] = df_completo['Restaurante'].fillna(
        df_completo['Id_Restaurante'].apply(lambda x: f"Restaurante {x}")
    )
    df_completo['Votaciones'] = df_completo['Votaciones'].fillna(0)
    df_completo['Valoracion'] = df_completo['Valoracion'].fillna(0.0)
    df_completo['Dirección'] = df_completo['Dirección'].fillna("")
    df_completo.to_csv(ARCHIVO_RESENAS, index=False)

    # analizar_todos usa checkpoint — solo procesara los restaurantes nuevos
    analizar_todos(df_completo)

    df_analisis = pd.read_csv(ARCHIVO_ANALISIS)
    df_ranking = pd.read_csv(ARCHIVO_RANKING, sep=";", skiprows=1)
    df_ranking = df_ranking[['Id_Restaurante', 'Restaurante', 'Votaciones', 'Valoracion', 'Dirección']]
    df_analisis['Id_Restaurante'] = df_analisis['Id_Restaurante'].astype(str)
    df_ranking['Id_Restaurante'] = df_ranking['Id_Restaurante'].astype(str)
    df = pd.merge(df_analisis, df_ranking, on='Id_Restaurante', how='left')
    df['Restaurante'] = df['Restaurante'].fillna(df['Id_Restaurante'].apply(lambda x: f"Restaurante {x}"))
    df['Votaciones'] = df['Votaciones'].fillna(0).astype(int)
    df['Valoracion'] = df['Valoracion'].fillna(0.0)
    df['Dirección'] = df['Dirección'].fillna("")
    df['platos_destacados'] = df['platos_destacados'].apply(parsear_platos)
    df['aspectos_positivos'] = df['aspectos_positivos'].apply(parsear_platos) if 'aspectos_positivos' in df.columns else df.get('aspectos_positivos', [[]]*len(df))
    df['aspectos_negativos'] = df['aspectos_negativos'].apply(parsear_platos) if 'aspectos_negativos' in df.columns else df.get('aspectos_negativos', [[]]*len(df))
    df['Id_Restaurante'] = df['Id_Restaurante'].astype(str)

    # Geocodificar direcciones (solo la primera vez, luego usa cache del CSV)
    df = geocodificar_restaurantes(df)

    coleccion = construir_chromadb(df)
    agente = construir_agente(df, coleccion)

    print("\n" + "="*60)
    print("  Proyecto 1 - Gemini + ChromaDB + LangGraph")
    print("  Escribe 'salir' para terminar")
    print("="*60)

    historial = []
    while True:
        pregunta = input("\nTu: ").strip()
        if pregunta.lower() in ['salir', 'exit', 'quit']:
            print("Hasta luego!")
            break
        if not pregunta:
            continue
        print(f"\nAsistente: {recomendar(agente, pregunta, historial)}")
