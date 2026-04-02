# ============================================================
# CABECERA
# ============================================================
# Alumno: Alejandro Vaquero
# URL Streamlit Cloud: https://...streamlit.app
# URL GitHub: https://github.com/...

# ============================================================
# IMPORTS
# ============================================================
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from openai import OpenAI
import json

# ============================================================
# CONSTANTES
# ============================================================
MODEL = "gpt-4.1-mini"

# -------------------------------------------------------
# >>> SYSTEM PROMPT — TU TRABAJO PRINCIPAL ESTÁ AQUÍ <<<
# -------------------------------------------------------
SYSTEM_PROMPT = """
Eres un analista de datos especializado en hábitos musicales de Spotify.
Tu única función es responder preguntas sobre los datos de escucha del usuario
generando código Python que produce una visualización con Plotly.

DATOS DISPONIBLES
=================
El DataFrame de pandas se llama `df` y ya está cargado en memoria.

Columnas disponibles:

  ts             (datetime) Fecha y hora de fin de reproducción (zona Europe/Madrid)
  ms_played      (int)      Milisegundos reproducidos
  seconds_played (float)    Segundos reproducidos
  minutes_played (float)    Minutos reproducidos
  hours_played   (float)    Horas reproducidas
  track          (str)      Nombre de la canción
  artist         (str)      Artista principal
  album          (str)      Álbum
  track_uri      (str)      URI único de Spotify
  reason_start   (str)      Motivo de inicio. Valores: {reason_start_values}
  reason_end     (str)      Motivo de fin. Valores: {reason_end_values}
  shuffle        (bool)     True si modo aleatorio estaba activo
  skipped        (bool)     True si la canción fue saltada
  platform       (str)      Plataforma. Valores: {plataformas}
  date           (date)     Fecha sin hora
  year           (int)      Año
  month          (int)      Mes (1-12)
  month_label    (str)      Mes en formato "YYYY-MM"
  weekday        (int)      Día de la semana (0=lunes, 6=domingo)
  weekday_name   (str)      Nombre del día en inglés
  hour           (int)      Hora del día (0-23)
  is_weekend     (bool)     True si es sábado o domingo
  semester       (str)      "H1" (enero-junio) o "H2" (julio-diciembre)
  season         (str)      Estación: Invierno, Primavera, Verano u Otoño

Rango de fechas: {fecha_min} a {fecha_max}

FORMATO DE RESPUESTA
====================
Responde SIEMPRE con un objeto JSON con exactamente estos tres campos:

Si la pregunta es sobre los datos musicales:
  {{"tipo": "grafico", "codigo": "<código Python>", "interpretacion": "<frase breve en español>"}}

Si la pregunta NO tiene relación con los datos musicales:
  {{"tipo": "fuera_de_alcance", "codigo": "", "interpretacion": "<explicación amable>"}}

Reglas:
- No incluyas ningún texto fuera del JSON.
- No uses bloques de código markdown (``` o ```python).
- Escapa los saltos de línea dentro de "codigo" como \\n.

INSTRUCCIONES PARA EL CÓDIGO
=============================
Las librerías disponibles son ÚNICAMENTE: df, pd, px, go.
NO importes ninguna librería adicional. NO uses matplotlib.

1. Crea siempre una variable llamada exactamente `fig`.
2. Usa px (plotly.express) para gráficos simples.
3. Pon siempre un título con el parámetro title=.
4. Rankings: px.bar(orientation="h"), ordenado de mayor a menor.
5. Evolución temporal: px.line() con eje X = month_label.
6. Distribución por hora: px.bar() con eje X = hour.
7. Comparaciones: px.bar() con color= para distinguir grupos.
8. Color principal de Spotify: color_discrete_sequence=["#1DB954"].
9. Para calcular horas usa la columna hours_played directamente.
10. Nunca uses valores hardcodeados: todo debe calcularse desde df.

TIPOS DE PREGUNTA QUE DEBES CUBRIR
====================================
A. Rankings y favoritos
   Top artistas/canciones por hours_played o número de reproducciones.
   Usar px.bar(orientation="h") ordenado de mayor a menor.

B. Evolución temporal
   px.line() o px.bar() con eje X = month_label, agrupando por mes.

C. Patrones de uso
   Horas del día: px.bar() con eje X = hour.
   Días de la semana: px.bar() ordenado lunes a domingo.
   Plataformas: px.pie() o px.bar().

D. Comportamiento de escucha
   Skips: px.pie() usando la columna skipped.
   Shuffle: px.pie() usando la columna shuffle.

E. Comparación entre períodos
   Usar semester o season con color= en px.bar() para comparar.

GUARDRAILS
==========
- No generes código que acceda a ficheros, redes o variables de entorno.
- No uses exec(), eval() ni importes librerías adicionales.
- Si piden datos inexistentes (géneros, valoraciones...), usa fuera_de_alcance.
- Nunca inventes datos.
"""


# ============================================================
# CARGA Y PREPARACIÓN DE DATOS
# ============================================================
@st.cache_data
def load_data():
    df = pd.read_json("streaming_history.json")

    # --- Renombrar columnas largas ---
    df = df.rename(columns={
        "master_metadata_track_name": "track",
        "master_metadata_album_artist_name": "artist",
        "master_metadata_album_album_name": "album",
        "spotify_track_uri": "track_uri",
    })

    # --- Convertir timestamp a datetime con zona horaria ---
    df["ts"] = pd.to_datetime(df["ts"], utc=True).dt.tz_convert("Europe/Madrid")

    # --- Columnas de duración ---
    df["seconds_played"] = df["ms_played"] / 1000
    df["minutes_played"] = df["ms_played"] / 60000
    df["hours_played"]   = df["ms_played"] / 3600000

    # --- Columnas derivadas de tiempo ---
    df["date"]         = df["ts"].dt.date
    df["year"]         = df["ts"].dt.year
    df["month"]        = df["ts"].dt.month
    df["month_label"]  = df["ts"].dt.strftime("%Y-%m")
    df["weekday"]      = df["ts"].dt.dayofweek
    df["weekday_name"] = df["ts"].dt.strftime("%A")
    df["hour"]         = df["ts"].dt.hour
    df["is_weekend"]   = df["weekday"].isin([5, 6])

    # --- Semestres y estaciones ---
    df["semester"] = df["month"].apply(lambda m: "H1" if m <= 6 else "H2")
    df["season"] = df["month"].map({
        12: "Invierno", 1: "Invierno", 2: "Invierno",
        3: "Primavera", 4: "Primavera", 5: "Primavera",
        6: "Verano",    7: "Verano",   8: "Verano",
        9: "Otoño",    10: "Otoño",   11: "Otoño",
    })

    # --- Normalizar skipped y shuffle ---
    df["skipped"] = df["skipped"].fillna(False).astype(bool)
    df["shuffle"] = df["shuffle"].astype(bool)

    # --- Filtrar reproducciones muy cortas (< 10 segundos) ---
    df = df[df["seconds_played"] >= 10].copy()

    return df


def build_prompt(df):
    fecha_min = df["ts"].min().strftime("%Y-%m-%d")
    fecha_max = df["ts"].max().strftime("%Y-%m-%d")
    plataformas = df["platform"].unique().tolist()
    reason_start_values = df["reason_start"].unique().tolist()
    reason_end_values = df["reason_end"].unique().tolist()

    return SYSTEM_PROMPT.format(
        fecha_min=fecha_min,
        fecha_max=fecha_max,
        plataformas=plataformas,
        reason_start_values=reason_start_values,
        reason_end_values=reason_end_values,
    )


# ============================================================
# FUNCIÓN DE LLAMADA A LA API
# ============================================================
def get_response(user_msg, system_prompt):
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_msg},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content


# ============================================================
# PARSING DE LA RESPUESTA
# ============================================================
def parse_response(raw):
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

    return json.loads(cleaned)


# ============================================================
# EJECUCIÓN DEL CÓDIGO GENERADO
# ============================================================
def execute_chart(code, df):
    local_vars = {"df": df, "pd": pd, "px": px, "go": go}
    exec(code, {}, local_vars)
    return local_vars.get("fig")


# ============================================================
# INTERFAZ STREAMLIT
# ============================================================
st.set_page_config(page_title="Spotify Analytics", layout="wide")

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.title("🔒 Acceso restringido")
    pwd = st.text_input("Contraseña:", type="password")
    if pwd:
        if pwd == st.secrets["PASSWORD"]:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Contraseña incorrecta.")
    st.stop()

st.title("🎵 Spotify Analytics Assistant")
st.caption("Pregunta lo que quieras sobre tus hábitos de escucha")

df = load_data()
system_prompt = build_prompt(df)

if prompt := st.chat_input("Ej: ¿Cuál es mi artista más escuchado?"):

    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analizando..."):
            try:
                raw = get_response(prompt, system_prompt)
                parsed = parse_response(raw)

                if parsed["tipo"] == "fuera_de_alcance":
                    st.write(parsed["interpretacion"])
                else:
                    fig = execute_chart(parsed["codigo"], df)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                        st.write(parsed["interpretacion"])
                        st.code(parsed["codigo"], language="python")
                    else:
                        st.warning("El código no produjo ninguna visualización. Intenta reformular la pregunta.")
                        st.code(parsed["codigo"], language="python")

            except json.JSONDecodeError:
                st.error("No he podido interpretar la respuesta. Intenta reformular la pregunta.")
            except Exception as e:
                st.error("Ha ocurrido un error al generar la visualización. Intenta reformular la pregunta.")


# ============================================================
# REFLEXIÓN TÉCNICA (máximo 30 líneas)
# ============================================================
#
# 1. ARQUITECTURA TEXT-TO-CODE
#    El LLM nunca ve los datos reales. Recibe únicamente el system prompt,
#    que describe la estructura del DataFrame (nombres de columnas, tipos y
#    rango de fechas), más la pregunta del usuario. A partir de eso genera
#    código Python como texto. Ese código se ejecuta en local con exec(),
#    usando el DataFrame real que ya está en memoria. El LLM no recibe los
#    datos directamente por dos razones: privacidad (son datos personales) y
#    coste (enviar 15.000 filas en cada llamada consumiría miles de tokens).
#
# 2. EL SYSTEM PROMPT COMO PIEZA CLAVE
#    El prompt le da al LLM tres cosas: el esquema exacto del DataFrame
#    (para que sepa qué columnas usar), el formato de salida obligatorio en
#    JSON (para que el parsing sea predecible), y guardrails que prohíben
#    importar librerías o acceder a ficheros.
#    Ejemplo que FUNCIONA gracias al prompt: "¿escucho más en shuffle o en
#    orden?" funciona porque el prompt especifica que shuffle es una columna
#    booleana y sugiere usar px.pie() para este tipo de pregunta.
#    Ejemplo que FALLARÍA sin el prompt: si quitase la instrucción de crear
#    siempre la variable fig, el LLM podría llamarla figure o chart,
#    y execute_chart() devolvería None porque busca exactamente fig.
#
# 3. EL FLUJO COMPLETO
#    Paso 1: Al arrancar, load_data() lee el JSON, renombra columnas,
#    convierte timestamps, crea columnas derivadas (hour, season, etc.) y
#    filtra reproducciones menores de 10 segundos. Se cachea en memoria.
#    Paso 2: build_prompt() inyecta en el system prompt los valores reales
#    del dataset (fechas, plataformas, valores de reason_start/end).
#    Paso 3: El usuario escribe una pregunta en el chat input.
#    Paso 4: get_response() envía el system prompt y la pregunta a la API
#    de OpenAI. El modelo devuelve un string que debería ser JSON.
#    Paso 5: parse_response() limpia posibles bloques markdown y convierte
#    el string en un diccionario Python con json.loads().
#    Paso 6: Si tipo es grafico, execute_chart() ejecuta el código con exec()
#    en un namespace que contiene df, pd, px y go. El código crea una figura
#    Plotly en la variable fig.
#    Paso 7: st.plotly_chart() renderiza la figura en pantalla, seguida de
#    la interpretacion como texto y el código generado como referencia.