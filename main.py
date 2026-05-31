"""
main.py
-------
Backend FastAPI para el sistema de recomendacion de restaurantes.
Expone un endpoint /recomendar que usa el Proyecto 1 (Gemini + ChromaDB + RAG).

Ejecucion: uvicorn main:app --reload
"""

import os
import ast
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

load_dotenv()

app = FastAPI(
    title="API Recomendacion Restaurantes Madrid",
    description="Sistema de recomendacion basado en Gemini + ChromaDB + RAG",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── MODELOS ────────────────────────────────────────────────────────────────────

class MensajeHistorial(BaseModel):
    role: str
    content: str

class ConsultaRequest(BaseModel):
    consulta: str
    historial: Optional[List[MensajeHistorial]] = []

class RecomendacionResponse(BaseModel):
    respuesta: str
    proyecto: str
    restaurantes: Optional[List[dict]] = []

# ── ESTADO GLOBAL ──────────────────────────────────────────────────────────────

agente_global = None
df_global = None

# ── INICIALIZAR ────────────────────────────────────────────────────────────────

def inicializar():
    global agente_global, df_global
    import proyecto1_gemini as p1

    def parsear_lista(v):
        if isinstance(v, list): return v
        try: return ast.literal_eval(v)
        except: return []

    print("Cargando datos...")
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    df_analisis = pd.read_csv(os.path.join(BASE_DIR, p1.ARCHIVO_ANALISIS))
    df_ranking  = pd.read_csv(os.path.join(BASE_DIR, p1.ARCHIVO_RANKING), sep=";", skiprows=1)
    df_ranking  = df_ranking[['Id_Restaurante', 'Restaurante', 'Votaciones', 'Valoracion', 'Dirección']]

    df = pd.merge(df_analisis, df_ranking, on='Id_Restaurante', how='left')
    df['platos_destacados']  = df['platos_destacados'].apply(parsear_lista)
    df['aspectos_positivos'] = df['aspectos_positivos'].apply(parsear_lista) if 'aspectos_positivos' in df.columns else [[]] * len(df)
    df['aspectos_negativos'] = df['aspectos_negativos'].apply(parsear_lista) if 'aspectos_negativos' in df.columns else [[]] * len(df)
    df['Id_Restaurante']     = df['Id_Restaurante'].astype(str)
    df['Restaurante']        = df['Restaurante'].fillna(df['Id_Restaurante'].apply(lambda x: f"Restaurante {x}"))
    df['Votaciones']         = df['Votaciones'].fillna(0).astype(int)
    df['Valoracion']         = df['Valoracion'].fillna(0.0)
    df['Dirección']          = df['Dirección'].fillna("")

    geo_path = os.path.join(BASE_DIR, p1.ARCHIVO_GEO)
    if os.path.exists(geo_path):
        df_geo = pd.read_csv(geo_path)[['Id_Restaurante', 'latitud', 'longitud']]
        df_geo['Id_Restaurante'] = df_geo['Id_Restaurante'].astype(str)
        df = df.merge(df_geo, on='Id_Restaurante', how='left')
    else:
        df['latitud']  = None
        df['longitud'] = None

    df_global = df

    print("Construyendo ChromaDB...")
    coleccion = p1.construir_chromadb(df)

    print("Construyendo agente...")
    agente = p1.construir_agente(df, coleccion)

    agente_global = lambda consulta, historial=[]: p1.recomendar(agente, consulta, historial)
    print("Backend listo!")

@app.on_event("startup")
async def startup_event():
    inicializar()

# ── ENDPOINTS ──────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "ok", "mensaje": "API Restaurantes Madrid funcionando"}

@app.get("/health")
def health():
    return {"status": "ok", "agente": agente_global is not None}

@app.post("/recomendar", response_model=RecomendacionResponse)
def recomendar(request: ConsultaRequest):
    if not request.consulta.strip():
        raise HTTPException(status_code=400, detail="La consulta no puede estar vacia")
    if agente_global is None:
        raise HTTPException(status_code=503, detail="El agente no esta inicializado")
    try:
        historial = [{"role": m.role, "content": m.content} for m in request.historial]
        respuesta = agente_global(request.consulta, historial)

        # Extraer nombres de restaurantes mencionados en la respuesta
        restaurantes = []
        if df_global is not None:
            for _, row in df_global.iterrows():
                nombre = str(row.get('Restaurante', ''))
                if nombre and nombre.lower() in respuesta.lower():
                    lat = row.get('latitud')
                    lon = row.get('longitud')
                    if pd.notna(lat) and pd.notna(lon):
                        restaurantes.append({
                            "nombre": nombre,
                            "latitud": float(lat),
                            "longitud": float(lon),
                            "valoracion": float(row.get('Valoracion', 0)),
                            "direccion": str(row.get('Dirección', '')),
                        })

        return RecomendacionResponse(
            respuesta=respuesta,
            proyecto="Gemini + ChromaDB + RAG",
            restaurantes=restaurantes,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/restaurantes")
def listar_restaurantes():
    """Devuelve todos los restaurantes con coordenadas."""
    try:
        if df_global is None:
            raise HTTPException(status_code=503, detail="Datos no cargados")
        cols = ['Id_Restaurante', 'Restaurante', 'Valoracion', 'Votaciones', 'Dirección', 'latitud', 'longitud']
        df = df_global[[c for c in cols if c in df_global.columns]].copy()
        df = df[df['latitud'].notna() & df['longitud'].notna()]
        return df.to_dict('records')
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
