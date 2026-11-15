import json
import os
from typing import Any

import firebase_admin
from fastapi import Depends, FastAPI, Header, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from firebase_admin import auth, credentials, firestore
from google import genai
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# 1. Validación e inicialización de variables de entorno (100% Cloud)
# ---------------------------------------------------------------------------
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
FIREBASE_CREDENTIALS_JSON = os.environ.get("FIREBASE_CREDENTIALS_JSON")
ALLOWED_USER_UID = os.environ.get("ALLOWED_USER_UID")

if not GEMINI_API_KEY:
    raise RuntimeError("Falta la variable de entorno GEMINI_API_KEY.")

if not FIREBASE_CREDENTIALS_JSON:
    raise RuntimeError("Falta la variable de entorno FIREBASE_CREDENTIALS_JSON.")

if not ALLOWED_USER_UID:
    raise RuntimeError("Falta la variable de entorno ALLOWED_USER_UID.")

# Parsear el JSON de credenciales de Firebase desde la variable de entorno
try:
    firebase_cred_dict = json.loads(FIREBASE_CREDENTIALS_JSON)
    # Normaliza saltos de línea escapados en private_key (común en paneles Cloud/GitHub)
    if isinstance(firebase_cred_dict.get("private_key"), str):
        firebase_cred_dict["private_key"] = firebase_cred_dict["private_key"].replace("\\n", "\n")
except json.JSONDecodeError as exc:
    raise RuntimeError(f"FIREBASE_CREDENTIALS_JSON no contiene un JSON válido: {exc}") from exc

# Inicializar Firebase Admin (una sola vez)
if not firebase_admin._apps:
      cred = credentials.Certificate(firebase_cred_dict)
      firebase_admin.initialize_app(cred)

db = firestore.client()

# Inicializar cliente oficial de Google GenAI
gemini_client = genai.Client(api_key=GEMINI_API_KEY)
MODEL_NAME = "gemini-2.5-flash"

# ---------------------------------------------------------------------------
# 2. Configuración de FastAPI y CORS
# ---------------------------------------------------------------------------
app = FastAPI(
    title="API Requisitos IA (FastAPI + Gemini + Firebase)",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# 3. Esquemas Pydantic
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    mensaje: str = Field(..., min_length=1, max_length=8000, description="Mensaje o consulta para Gemini")


class ChatResponse(BaseModel):
    uid: str
    modelo: str
    mensaje: str
    respuesta: str
    registro_id: str


# ---------------------------------------------------------------------------
# 4. Dependencia de seguridad: Validar Firebase Auth Token y UID permitido
# ---------------------------------------------------------------------------
def verify_allowed_user(authorization: str | None = Header(default=None)) -> dict[str, Any]:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Falta el header Authorization con formato 'Bearer <firebase_id_token>'.",
        )

    id_token = authorization.split("Bearer ", 1)[1].strip()
    if not id_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="El token Bearer está vacío.",
        )

    try:
        decoded_token = auth.verify_id_token(id_token)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Token de Firebase inválido o expirado: {exc}",
        ) from exc

    token_uid = decoded_token.get("uid")
    if token_uid != ALLOWED_USER_UID.strip():
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Acceso denegado: este UID no está autorizado para usar el servicio.",
        )

    return decoded_token


# ---------------------------------------------------------------------------
# 5. Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok", "modelo": MODEL_NAME}


@app.post("/chat", response_model=ChatResponse)
def chat_with_gemini(
    payload: ChatRequest,
    user: dict[str, Any] = Depends(verify_allowed_user),
) -> ChatResponse:
    uid = user["uid"]

    # 1) Consultar a Gemini usando el SDK oficial google-genai
    try:
        gemini_response = gemini_client.models.generate_content(
            model=MODEL_NAME,
            contents=payload.mensaje,
        )
        reply_text = (gemini_response.text or "").strip()
        if not reply_text:
            raise ValueError("Gemini devolvió una respuesta vacía.")
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Error al consultar Gemini ({MODEL_NAME}): {exc}",
        ) from exc

    # 2) Guardar la interacción en Firestore bajo usuarios/{uid}/historial
    try:
        historial_ref = db.collection("usuarios").document(uid).collection("historial")
        _, doc_ref = historial_ref.add(
            {
                "uid": uid,
                "email": user.get("email", ""),
                "mensaje": payload.mensaje,
                "respuesta": reply_text,
                "modelo": MODEL_NAME,
                "creado_en": firestore.SERVER_TIMESTAMP,
            }
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al guardar la conversación en Firestore: {exc}",
        ) from exc

    return ChatResponse(
        uid=uid,
        modelo=MODEL_NAME,
        mensaje=payload.mensaje,
        respuesta=reply_text,
        registro_id=doc_ref.id,
    )
